---
name: data-and-secrets-hygiene
description: Rules that keep proprietary experimental data, model checkpoints, credentials and API keys out of the repository and out of external services. Use before every commit, whenever you handle data files, tokens, .env files, notebooks with outputs, or are asked to upload, share, publish or push anything.
---

# Data and secrets hygiene

## Hard rules

1. Never commit: experimental/instrument data (`*.nd2 *.czi *.lif *.tif` stacks,
   spreadsheets, `*.h5 *.npy *.mat` …), model weights (`*.pt *.pth *.safetensors`),
   credentials (`.env`, keys, tokens, service-account JSON).
2. Never paste a secret into code, docs, notebooks, commit messages, issues or
   chat. Read secrets from environment variables or the OS keychain at runtime
   (`HF_TOKEN` is read by `huggingface_hub`; never hard-code it).
3. Never upload repository content or data to any external service, and never
   push, publish, open a public repo, or change repository visibility without
   explicit human approval for that specific action.
4. Keep real data **outside** the repo (e.g. `$SAM3_DATA_DIR`). Tests use only
   synthetic data or the public assets already in `assets/`.

## Before committing

```bash
python tools/security/scan_sensitive.py --staged
git config core.hooksPath .githooks   # once per clone; runs the scan automatically
```

* Clear notebook outputs before committing (`jupyter nbconvert --clear-output --inplace nb.ipynb`):
  outputs can contain images of proprietary data and file paths.
* False positive? Add a justified entry to `tools/security/allowlist.txt` or put
  `sensitive-scan: allow` on the line, and say so in the PR for human review.

## If a secret or data file was committed

Stop. Tell the human immediately. **Rotate the credential first** — removing it
from history does not un-leak it. History rewriting (`git filter-repo`) is a
human decision. See `docs/SECURITY_AND_DATA.md`.
