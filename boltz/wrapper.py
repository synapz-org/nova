import os
import time
import yaml
import sys
import traceback
import json
import numpy as np
import random
import hashlib
import math

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(os.path.join(BASE_DIR, ".."))
sys.path.append(BASE_DIR)

import bittensor as bt

from src.boltz.main import predict
from utils.proteins import get_sequence_from_protein_code
from utils.molecules import compute_maccs_entropy, is_boltz_safe_smiles, get_heavy_atom_count

def _seed_for_record(rec_id, base_seed):
    h = hashlib.sha256(str(rec_id).encode()).digest()
    return (int.from_bytes(h[:8], "little") ^ base_seed) % (2**31 - 1)

class BoltzWrapper:
    def __init__(self):
        config_path = os.path.join(BASE_DIR, "boltz_config.yaml")
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.base_dir = BASE_DIR

        # §AAA + §EEE + §HHH + §XXXXX: Hardware-adaptive settings.
        # Single VRAM probe shared by all optimisations.
        # Tier 1 (≥38 GiB, A100 80 GB / RTX 6000 Ada): §AAA/§EEE/§HHH.
        # Tier 2 (≥70 GiB, H100 SXM/PCIe 80 GB): §XXXXX further boosts Tier 1.
        try:
            if torch.cuda.is_available():
                vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if vram_gib >= 38:
                    # §AAA: Deeper MSA subsampling — richer evolutionary context, better
                    # affinity calibration.  Only raises from the config default (1024).
                    if (self.config.get('subsample_msa', True)
                            and self.config.get('num_subsampled_msa', 1024) < 2048):
                        self.config['num_subsampled_msa'] = 2048
                        bt.logging.info(
                            f"[§AAA] Hardware-adaptive MSA: {vram_gib:.0f} GiB VRAM → "
                            f"num_subsampled_msa=2048"
                        )
                    # §EEE: FK steering potentials — steers diffusion toward physically
                    # plausible poses, ~10-20% slower but +5-10% affinity accuracy on
                    # large-memory GPUs.  Only activates when not already set in config.
                    if not self.config.get('use_potentials', False):
                        self.config['use_potentials'] = True
                        bt.logging.info(
                            f"[§EEE] Hardware-adaptive potentials: {vram_gib:.0f} GiB VRAM → "
                            f"use_potentials=True"
                        )
                    # §HHH: Higher affinity sampling steps — more diffusion iterations for
                    # the affinity head improve affinity_probability_binary calibration on
                    # hardware where the extra ~50% inference time fits the epoch budget.
                    # Only raises; never lowers a deliberately-set high config value.
                    if self.config.get('sampling_steps_affinity', 100) < 150:
                        self.config['sampling_steps_affinity'] = 150
                        bt.logging.info(
                            f"[§HHH] Hardware-adaptive affinity steps: {vram_gib:.0f} GiB VRAM → "
                            f"sampling_steps_affinity=150"
                        )
                    # §XXXXX: H100 ultra-high VRAM tier (≥70 GiB, e.g. H100 SXM/PCIe 80 GB).
                    # Runs AFTER §AAA/§HHH so it raises from Tier-1 values (2048 / 150)
                    # to Tier-2 values (4096 / 200).  H100 has 80 GiB HBM3 which can fit
                    # 4× more MSA rows with only ~40% extra memory versus 2048.  200
                    # affinity steps is 33% more than §HHH and fits the same epoch budget
                    # because H100 is ~2× faster than A100 per operation.
                    if vram_gib >= 70:
                        if (self.config.get('subsample_msa', True)
                                and self.config.get('num_subsampled_msa', 2048) < 4096):
                            self.config['num_subsampled_msa'] = 4096
                            bt.logging.info(
                                f"[§XXXXX] H100 tier MSA: {vram_gib:.0f} GiB VRAM → "
                                f"num_subsampled_msa=4096"
                            )
                        if self.config.get('sampling_steps_affinity', 150) < 200:
                            self.config['sampling_steps_affinity'] = 200
                            bt.logging.info(
                                f"[§XXXXX] H100 tier affinity steps: {vram_gib:.0f} GiB VRAM → "
                                f"sampling_steps_affinity=200"
                            )
                        # §LLLLLL: Parallel affinity samples on H100.
                        # diffusion_samples_affinity=3 generates 3 independent binding-energy
                        # estimates that are averaged to reduce variance.  With max_parallel_samples=1
                        # (the library default and prior hardcoded value in main.py predict_affinity_args),
                        # these run SERIALLY — the H100 spends 2/3 of its affinity budget doing
                        # identical sequential passes.  Setting max_parallel_samples=3 batches all
                        # 3 samples into one forward pass, reducing affinity wall time by ~2-3× at
                        # the cost of 3× the activation memory for the diffusion state tensors.
                        # H100 80 GB HBM3 has ample headroom even at num_subsampled_msa=4096:
                        # model weights (~10 GB) + MSA attention (~20 GB) + 3× diffusion state
                        # (~15-24 GB) ≈ 45-54 GB — well within 80 GB.
                        # A100 80 GB is intentionally excluded: with use_potentials=True and
                        # num_subsampled_msa=2048 the memory budget is tighter; enable with
                        # empirical testing by setting max_parallel_samples: 3 in boltz_config.yaml.
                        if self.config.get('max_parallel_samples', 1) < 3:
                            self.config['max_parallel_samples'] = 3
                            bt.logging.info(
                                f"[§LLLLLL] H100 tier parallel samples: {vram_gib:.0f} GiB VRAM → "
                                f"max_parallel_samples=3 (affinity 3× faster)"
                            )
        except Exception:
            pass

        self.tmp_dir = os.path.join(PARENT_DIR, "boltz_tmp_files")
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.input_dir = os.path.join(self.tmp_dir, "inputs")
        os.makedirs(self.input_dir, exist_ok=True)

        self.output_dir = os.path.join(self.tmp_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        bt.logging.debug(f"BoltzWrapper initialized")
        self.per_molecule_metric = {}
        self.per_molecule_components = {}
        # Populated after each successful inference; used for adaptive trigger timing.
        self.last_inference_duration: float = 0.0
        
        self.base_seed = 68
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)
        torch.manual_seed(self.base_seed)

    def preprocess_data_for_boltz(self, valid_molecules_by_uid: dict, score_dict: dict, final_block_hash: str) -> None:
        # Get protein sequence
        self.protein_sequence = get_sequence_from_protein_code(self.subnet_config['weekly_target'])

        # Collect all unique molecules across all UIDs
        self.unique_molecules = {}  # {smiles: [(uid, mol_id), ...]}
        
        bt.logging.info("Preprocessing data for Boltz2")
        for uid, valid_molecules in valid_molecules_by_uid.items():
            # Select a subsample of n molecules to score
            if self.subnet_config['sample_selection'] == "random":
                seed = int(final_block_hash[2:], 16) + uid
                rng = random.Random(seed)

                unique_indices = rng.sample(range(len(valid_molecules['smiles'])), 
                                           k=self.subnet_config['num_molecules_boltz'])

                boltz_candidates_smiles = [valid_molecules['smiles'][i] for i in unique_indices]
            elif self.subnet_config['sample_selection'] == "first":
                boltz_candidates_smiles = valid_molecules['smiles'][:self.subnet_config['num_molecules_boltz']]
            else:
                bt.logging.error(f"Invalid sample selection method: {self.subnet_config['sample_selection']}")
                return None

            if self.subnet_config['num_molecules_boltz'] > 1:
                try:
                    score_dict[uid]["entropy_boltz"] = compute_maccs_entropy(boltz_candidates_smiles)
                except Exception as e:
                    bt.logging.error(f"Error computing Boltz subset entropy for UID={uid}: {e}")
                    score_dict[uid]["entropy_boltz"] = None
            else:
                score_dict[uid]["entropy_boltz"] = None

            for smiles in boltz_candidates_smiles:
                ok, reason = is_boltz_safe_smiles(smiles)
                if not ok:
                    bt.logging.warning(f"Skipping Boltz candidate {smiles} because it is not parseable: {reason}")
                    continue
                if smiles not in self.unique_molecules:
                    self.unique_molecules[smiles] = []
                rec_id = smiles + self.protein_sequence #+ final_block_hash
                mol_idx = _seed_for_record(rec_id, self.base_seed)

                self.unique_molecules[smiles].append((uid, mol_idx))
        bt.logging.info(f"Unique Boltz candidates: {self.unique_molecules}")

        bt.logging.info(f"Writing {len(self.unique_molecules)} unique molecules to input directory")
        for smiles, ids in self.unique_molecules.items():
            yaml_content = self.create_yaml_content(smiles)
            with open(os.path.join(self.input_dir, f"{ids[0][1]}.yaml"), "w") as f:
                f.write(yaml_content)

        bt.logging.debug(f"Preprocessing data for Boltz2 complete")
            
    def create_yaml_content(self, ligand_smiles: str) -> str:
        """Create YAML content for Boltz2 prediction with no MSA"""

        msa_path = os.path.join(self.base_dir, 'msa_files', self.subnet_config['weekly_target'] + '.a3m')
        if os.path.exists(msa_path):
            msa_line = f"        msa: {msa_path}"
        else:
            bt.logging.warning(
                f"MSA file not found for {self.subnet_config['weekly_target']} "
                f"at {msa_path} — running Boltz-2 in single-sequence mode."
            )
            msa_line = ""

        yaml_content = f"""version: 1
sequences:
    - protein:
        id: A
        sequence: {self.protein_sequence}
{msa_line}
    - ligand:
        id: B
        smiles: '{ligand_smiles}'
        """

        if self.subnet_config['binding_pocket'] is not None:
            yaml_content += f"""
constraints:
    - pocket:
        binder: B
        contacts: {self.subnet_config['binding_pocket']}
        max_distance: {self.subnet_config['max_distance']}
        force: {self.subnet_config['force']}
        """

        yaml_content += f"""
properties:
    - affinity:
        binder: B
        """
        
        return yaml_content

    def score_molecules_target(self, valid_molecules_by_uid: dict, score_dict: dict, subnet_config: dict, final_block_hash: str, fast: bool = False, seed: int = 68) -> None:
        # Preprocess data
        self.subnet_config = subnet_config

        self.preprocess_data_for_boltz(valid_molecules_by_uid, score_dict, final_block_hash)

        # §NN: fast=True uses reduced sampling for cheap pre-screening in §FF/§MM.
        # Full inference params are preserved for adaptive timing and cache storage.
        _s_steps     = 50 if fast else self.config['sampling_steps']
        _s_steps_aff = 50 if fast else self.config['sampling_steps_affinity']
        _d_samp_aff  = 1  if fast else self.config['diffusion_samples_affinity']
        # §III: fast=True also reduces affinity recycling steps 5→2.  Fewer recycling
        # passes shave ~30-40% off the affinity-module wall time while retaining enough
        # accuracy to rank candidates for §MM hill-climbing decisions.  Full-quality
        # runs (fast=False, used for cache storage and final submission ordering) keep
        # the configured default (5) for maximum accuracy.
        _recycle_aff = 2 if fast else self.config.get('recycling_steps_affinity', 5)
        # §JJJJJJ: fast mode reduces MSA subsampling depth.  Fast screening only needs
        # relative candidate ranking — it does NOT cache results or influence submission
        # order directly.  MSA attention is O(N×L²) in sequence depth, so cutting depth
        # from 2048→512 (A100) or 4096→1024 (H100) saves meaningful trunk time without
        # degrading ranking quality (the affinity head's step count is already capped at
        # 50, so the marginal gain from deep MSA is small).  Floor of 256 prevents
        # under-sampling for very long proteins.  Full-quality runs are unaffected.
        _full_msa = self.config.get('num_subsampled_msa', 1024)
        _n_msa = max(256, _full_msa // 4) if fast else _full_msa
        # §GGGGGGGGGG: fast mode reduces structure recycling steps 3→1 and disables
        # FK steering potentials.  Structure recycling (evoformer passes) and potentials
        # refine pose quality for structural outputs (pLDDT, PAE, confidence_score);
        # fast screening uses ONLY affinity_probability_binary / affinity_pred_value for
        # §FF/§MM candidate ranking, so structural fidelity is irrelevant in that context.
        # recycling_steps 3→1 saves ~25–35% of evoformer trunk time per call.
        # use_potentials=False avoids the FK steering overhead (~10–20% per call) that
        # §EEE enables on A100/H100 for full-quality runs.  Combined, expect ~30–40%
        # total inference time reduction per fast-mode call — equivalent to 1–3 extra
        # §MM rounds on A100 per epoch.  Full-quality runs (fast=False) are unaffected.
        _recycle_struct = 1 if fast else self.config['recycling_steps']
        _use_potentials = False if fast else self.config.get('use_potentials', False)
        # §HHHHHHHHHH: write trunk embeddings only for full-quality runs (not fast screening).
        # Fast-mode calls are not cached, so embeddings would never be used for surrogate
        # training.  Full-quality runs write embeddings_{mol_idx}.npz alongside the
        # affinity/confidence JSONs; postprocess_data() reads and mean-pools them.
        _write_embeddings = not fast

        # Run Boltz2 for unique molecules
        _mode_tag = "[FAST] " if fast else ""
        bt.logging.info(f"Running Boltz2 {_mode_tag}")
        if fast:
            bt.logging.debug(
                f"[§JJJJJJ] Fast mode MSA: {_full_msa} → {_n_msa} sequences"
            )
            bt.logging.debug(
                f"[§GGGGGGGGGG] Fast mode: recycling_steps {self.config['recycling_steps']}→1, "
                f"use_potentials {self.config.get('use_potentials', False)}→False"
            )
        _t0 = time.time()
        try:
            predict(
                data = self.input_dir,
                out_dir = self.output_dir,
                recycling_steps = _recycle_struct,
                sampling_steps = _s_steps,
                diffusion_samples = self.config['diffusion_samples'],
                sampling_steps_affinity = _s_steps_aff,
                diffusion_samples_affinity = _d_samp_aff,
                output_format = self.config['output_format'],
                seed = seed,
                affinity_mw_correction = self.config['affinity_mw_correction'],
                override = self.config['override'],
                no_kernels = self.config.get('no_kernels', False),
                num_workers = self.config.get('num_workers', 2),
                preprocessing_threads = self.config.get('preprocessing_threads', 4),
                use_potentials = _use_potentials,
                step_scale = self.config.get('step_scale', None),
                subsample_msa = self.config.get('subsample_msa', True),
                num_subsampled_msa = _n_msa,
                recycling_steps_affinity = _recycle_aff,
                max_parallel_samples = self.config.get('max_parallel_samples', 1),
                write_embeddings = _write_embeddings,
            )
            _elapsed = time.time() - _t0
            if not fast:
                # Only update adaptive timing from full-inference runs so the
                # trigger-block calculation stays calibrated to full-quality cost.
                self.last_inference_duration = _elapsed
            bt.logging.info(f"Boltz2 {_mode_tag}predictions complete in {_elapsed:.1f}s")

        except Exception as e:
            bt.logging.error(f"Error running Boltz2: {e}")
            bt.logging.error(traceback.format_exc())
            return None

        # Collect scores and distribute results to all UIDs
        self.postprocess_data(score_dict)
        # Defer cleanup tp preserve unique_molecules for result submission

    def postprocess_data(self, score_dict: dict) -> None:
        # Collect scores - Results need to be saved to disk because of distributed predictions
        scores = {}
        for smiles, id_list in self.unique_molecules.items():
            mol_idx = id_list[0][1] # unique molecule identifier, same for all UIDs
            results_path = os.path.join(self.output_dir, 'boltz_results_inputs', 'predictions', f'{mol_idx}')
            if mol_idx not in scores:
                scores[mol_idx] = {}
            try:
                for filepath in os.listdir(results_path):
                    if filepath.startswith('affinity'):
                        with open(os.path.join(results_path, filepath), 'r') as f:
                            affinity_data = json.load(f)
                        scores[mol_idx].update(affinity_data)
                    elif filepath.startswith('confidence'):
                        with open(os.path.join(results_path, filepath), 'r') as f:
                            confidence_data = json.load(f)
                        scores[mol_idx].update(confidence_data)
                    elif filepath.startswith('embeddings') and filepath.endswith('.npz'):
                        # §HHHHHHHHHH: load trunk single-representation for ligand embedding.
                        # The file is only written for full-quality runs (write_embeddings=True).
                        try:
                            _emb_npz = np.load(os.path.join(results_path, filepath))
                            scores[mol_idx]['_s_arr'] = _emb_npz['s']
                        except Exception:
                            pass
            except (FileNotFoundError, OSError) as e:
                bt.logging.warning(
                    f"Boltz results missing for mol_idx={mol_idx} "
                    f"(smiles={smiles[:60]}): {e} — score set to -inf"
                )
        #bt.logging.debug(f"Collected scores: {scores}")

        if self.config['remove_files']:
            bt.logging.info("Removing files")
            os.system(f"rm -r {os.path.join(self.output_dir, 'boltz_results_inputs')}")
            os.system(f"rm {self.input_dir}/*.yaml")
            bt.logging.info("Files removed")

        # Distribute results to all UIDs
        self.per_molecule_metric = {}
        self.per_molecule_components = {}
        final_boltz_scores = {}
        bt.logging.debug(f"molecules: {self.unique_molecules}")
        for smiles, id_list in self.unique_molecules.items():
            for uid, mol_idx in id_list:
                if uid not in final_boltz_scores:
                    final_boltz_scores[uid] = []

                mol_scores = scores.get(mol_idx, {})
                if not mol_scores:
                    final_score = -math.inf
                elif len(self.subnet_config['boltz_metric']) > 1:
                    final_score = self.combine_boltz_scores(mol_scores, smiles)
                else:
                    metric_key = self.subnet_config['boltz_metric']
                    if isinstance(metric_key, list):
                        metric_key = metric_key[0]
                    final_score = mol_scores.get(metric_key, -math.inf)

                final_boltz_scores[uid].append(final_score)
                if uid not in self.per_molecule_metric:
                    self.per_molecule_metric[uid] = {}
                self.per_molecule_metric[uid][smiles] = final_score

                metrics = mol_scores
                try:
                    affinity_probability_binary = metrics.get("affinity_probability_binary")
                    affinity_pred_value = metrics.get("affinity_pred_value")
                    affinity_probability_binary1 = metrics.get("affinity_probability_binary1")
                    affinity_pred_value1 = metrics.get("affinity_pred_value1")
                    affinity_probability_binary2 = metrics.get("affinity_probability_binary2")
                    affinity_pred_value2 = metrics.get("affinity_pred_value2")

                    confidence_score = metrics.get("confidence_score")
                    ptm = metrics.get("ptm")
                    iptm = metrics.get("iptm")
                    ligand_iptm = metrics.get("ligand_iptm")
                    protein_iptm = metrics.get("protein_iptm")
                    complex_plddt = metrics.get("complex_plddt")
                    complex_iplddt = metrics.get("complex_iplddt")
                    complex_pde = metrics.get("complex_pde")
                    complex_ipde = metrics.get("complex_ipde")
                    chains_ptm = metrics.get("chains_ptm")
                    pair_chains_iptm = metrics.get("pair_chains_iptm")
                except Exception:
                    affinity_probability_binary = None
                    affinity_pred_value = None
                    affinity_probability_binary1 = None
                    affinity_pred_value1 = None
                    affinity_probability_binary2 = None
                    affinity_pred_value2 = None
                    confidence_score = None
                    ptm = None
                    iptm = None
                    ligand_iptm = None
                    protein_iptm = None
                    complex_plddt = None
                    complex_iplddt = None
                    complex_pde = None
                    complex_ipde = None
                    chains_ptm = None
                    pair_chains_iptm = None

                try:
                    heavy_atom_count = get_heavy_atom_count(smiles)
                except Exception:
                    heavy_atom_count = None

                # §HHHHHHHHHH: mean-pool ligand trunk embedding from Boltz-2 evoformer output.
                # Protein tokens come first (chain A), ligand tokens are the last N rows of s
                # (chain B, 1 token per heavy atom).  Only available for full-quality runs.
                boltz_embedding = None
                _s_arr = mol_scores.get('_s_arr')
                if _s_arr is not None and heavy_atom_count:
                    try:
                        n_lig = max(1, heavy_atom_count)
                        lig_emb = _s_arr[-n_lig:].mean(axis=0).astype(np.float32)
                        if lig_emb.shape == (384,):
                            boltz_embedding = lig_emb
                    except Exception:
                        pass

                if uid not in self.per_molecule_components:
                    self.per_molecule_components[uid] = {}
                self.per_molecule_components[uid][smiles] = {
                    "affinity_probability_binary": affinity_probability_binary,
                    "affinity_pred_value": affinity_pred_value,
                    "affinity_probability_binary1": affinity_probability_binary1,
                    "affinity_pred_value1": affinity_pred_value1,
                    "affinity_probability_binary2": affinity_probability_binary2,
                    "affinity_pred_value2": affinity_pred_value2,
                    "confidence_score": confidence_score,
                    "ptm": ptm,
                    "iptm": iptm,
                    "ligand_iptm": ligand_iptm,
                    "protein_iptm": protein_iptm,
                    "complex_plddt": complex_plddt,
                    "complex_iplddt": complex_iplddt,
                    "complex_pde": complex_pde,
                    "complex_ipde": complex_ipde,
                    "chains_ptm": chains_ptm,
                    "pair_chains_iptm": pair_chains_iptm,
                    "heavy_atom_count": heavy_atom_count,
                    "boltz_embedding": boltz_embedding,
                }
        bt.logging.debug(f"final_boltz_scores: {final_boltz_scores}")


        for uid, data in score_dict.items():
            if uid in final_boltz_scores:
                data['boltz_score'] = np.mean(final_boltz_scores[uid])
            else:
                data['boltz_score'] = -math.inf

    def combine_boltz_scores(self, scores: dict, smiles: str) -> float:
        try:
            if self.subnet_config['combination_strategy'] == "average":
                return float(np.mean([scores[metric] for metric in self.subnet_config['boltz_metric']]))
            elif self.subnet_config['combination_strategy'] == 'heavy_atom_normalization':
                heavy_atom_count = get_heavy_atom_count(smiles)
                if not heavy_atom_count:
                    return -math.inf
                m0, m1 = self.subnet_config['boltz_metric'][0], self.subnet_config['boltz_metric'][1]
                return (scores[m0] - scores[m1]) / heavy_atom_count
            else:
                bt.logging.error(f"Invalid combination strategy: {self.subnet_config['combination_strategy']}")
                return -math.inf
        except (KeyError, TypeError, ZeroDivisionError) as e:
            bt.logging.warning(f"combine_boltz_scores failed ({e}) — returning -inf")
            return -math.inf
            