"""Proxy scoring for ranking molecules by estimated validator score."""


class ProxyScorer:
    @staticmethod
    def rank_by_proxy(scored_molecules):
        """Rank by affinity / heavy_atom_count.

        Input: list of (name, smiles, affinity, heavy_atoms)
        Output: same tuples with proxy score appended, sorted descending by proxy.
        """
        results = []
        for mol in scored_molecules:
            name, smiles, affinity, heavy_atoms = mol
            if heavy_atoms == 0:
                proxy = float("-inf")
            else:
                proxy = affinity / heavy_atoms
            results.append((name, smiles, affinity, heavy_atoms, proxy))
        results.sort(key=lambda x: x[4], reverse=True)
        return results

    @staticmethod
    def rank_with_antitarget(molecules, antitarget_weight=0.9):
        """Rank by (target - weight*antitarget) / heavy_atoms.

        Input: list of (name, smiles, target_score, antitarget_score, heavy_atoms)
        Output: same tuples with proxy score appended, sorted descending by proxy.
        Handle ha=0 by using float('-inf') as proxy.
        """
        results = []
        for mol in molecules:
            name, smiles, target_score, antitarget_score, heavy_atoms = mol
            if heavy_atoms == 0:
                proxy = float("-inf")
            else:
                proxy = (target_score - antitarget_weight * antitarget_score) / heavy_atoms
            results.append((name, smiles, target_score, antitarget_score, heavy_atoms, proxy))
        results.sort(key=lambda x: x[5], reverse=True)
        return results
