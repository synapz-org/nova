# `.gitattributes` `* text eol=lf` corrupts binary `.pt` files

## Symptom
After `git checkout`, `.pt` files (PyTorch state dicts) fail to load — `torch.load(...)` raises `UnpicklingError`, "invalid load key", or returns garbage tensors. `git status` may show the `.pt` file as modified even though you didn't touch it.

## Cause
The repo's `.gitattributes` declares `* text eol=lf`. Git treats all files as text and applies CRLF→LF normalization on checkout. For real text files this is harmless; for binary `.pt` files, any bytes that happen to match `\r\n` get rewritten to `\n`, silently corrupting the model.

## Handling
Mark the specific binary paths as not-text in `.git/info/attributes` (local, not committed) before touching them:

```sh
echo "PSICHIC/trained_weights/PDBv2020_PSICHIC/model.pt -text" >> .git/info/attributes
echo "path/to/your/binary.pt -text" >> .git/info/attributes
git checkout -- path/to/your/binary.pt   # re-fetch unmangled
```

Or fix at the source: add `*.pt binary` to the repo's `.gitattributes` (would need an upstream PR).

If a `.pt` file already shows as modified on a fresh clone, that's the smoking gun — the normalization fired on checkout.
