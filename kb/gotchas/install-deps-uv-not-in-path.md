# install_deps.sh fails silently on a fresh box because `uv` isn't on PATH

## Symptom
`./install_deps.sh` runs for several minutes (Rust crate compilation succeeds, external tools download fine), then exits 0 with no `.venv/` created. Tail of the log shows:

```
./wasm_build_py.sh: line 2: maturin: command not found
./install_deps.sh: line 84: cd: external_tools/timelock/py: No such file or directory
./install_deps.sh: line 85: uv: command not found
```

## Cause
The script installs `uv` via `wget -qO- https://astral.sh/uv/install.sh | sh` at line 30, which drops the binary at `~/.local/bin/uv`. Line 33 has `source "$HOME/.local/bin/env"` **commented out**:

```bash
# Install uv:
wget -qO- https://astral.sh/uv/install.sh | sh
#source $HOME/.local/bin/env   ← this would add ~/.local/bin to PATH
```

So when the script later tries `uv venv ...` (line 78) on a fresh shell, `uv` isn't on PATH. The error is silenced by chained `&&` short-circuiting through the rest of the install block.

## Handling
Either:
- Uncomment the `source $HOME/.local/bin/env` line in install_deps.sh, OR
- Run the install with `export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH` set first

Or finish the install manually after the script bails:

```bash
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
cd /home/ubuntu/nova
uv venv --python python3
source .venv/bin/activate
uv pip install -r requirements/requirements.txt
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install patchelf maturin==1.8.3
cd external_tools/boltz && uv pip install -e . && cd ../..
cd external_tools/boltzgen && uv pip install -e . && cd ../..
# ... and so on per the install_deps.sh script
```

## Detection
The reported exit code is **0** (no failure), but `ls .venv/bin/python` returns nothing. A working install has that path.

A reasonable post-install check: `python3 -c "import boltz, boltzgen, bittensor; print('ok')"` against `.venv/bin/python` — if any import fails, the install is incomplete.
