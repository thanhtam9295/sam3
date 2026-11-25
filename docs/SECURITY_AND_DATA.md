# Security, credentials and proprietary data

## 1. What must never enter the repository

| Category | Examples | Guard |
|---|---|---|
| Credentials | `.env`, API keys (HF, OpenAI, Anthropic, Azure, AWS, GitHub), private keys, service-account JSON, OAuth secrets | `.gitignore`, `scan_sensitive.py` secret patterns, Claude deny rules |
| Experimental data | microscopy/instrument files (`.nd2 .czi .lif .lsm .oib .ims .vsi .svs .ndpi .dcm .nii .fcs`…), spreadsheets, `.h5 .mat .npy .parquet`, annotation exports of private data | `.gitignore`, forbidden-type rule |
| Model weights | `.pt .pth .ckpt .safetensors .onnx .bin` | `.gitignore`, forbidden-type rule |
| Large files | anything > 5 MB not in `tools/security/allowlist.txt` | size rule |
| Notebook outputs | rendered images of data, absolute paths, tokens in cell output | clear outputs before commit |

Keep data outside the repo (e.g. set `SAM3_DATA_DIR`) and pass paths at runtime.

## 2. Automated guards

* `.githooks/pre-commit` (enable: `git config core.hooksPath .githooks`):
  `scan_sensitive.py --staged`, claim check, validation warning.
* CI `guards` job: `scan_sensitive.py --all` and `--history` (full history),
  so a bypassed hook is still caught before merge.
* Recommended in addition (T-007): GitHub secret scanning + push protection
  (or GitHub Advanced Security on Enterprise) and a pinned `gitleaks` job.

Run anytime:

```bash
python tools/security/scan_sensitive.py --all
python tools/security/scan_sensitive.py --history     # needs a full (non-shallow) clone
```

Audit on 2026-11-25: tree clean; 50-commit local history clean (shallow clone,
KI-014); one size exception allow-listed (`assets/dog.gif`, public upstream asset).

## 3. Secrets at runtime

* Hugging Face: the user runs `hf auth login` (token stored in the user profile,
  outside the repo) or sets `HF_TOKEN` in their shell. Code never reads or logs it.
* Service deployments: secrets in the platform's secret store (Azure Key Vault,
  GitHub Actions secrets/environments), injected as environment variables.
* `.env.example` files may document variable *names* only.

## 4. Data egress rules (no transfer without approval)

* Agents and code must not upload repository content, images, predictions or
  logs to external services without explicit human approval.
* **SAM 3 Agent** (`sam3/agent`) sends images to its LLM endpoint (KI-013): use
  a locally hosted or enterprise-approved endpoint only.
* `huggingface_hub` downloads checkpoints (inbound only). Set `HF_HUB_OFFLINE=1`
  with a local `--checkpoint` for fully offline operation; tests run offline.
* The REST service processes images in memory and does not persist or log them.
* AI coding agents: do not paste data samples or secrets into prompts; prefer
  enterprise tenants of each tool with data-retention controls (see
  ENTERPRISE_MIGRATION.md §5).

## 5. If something sensitive was committed or pushed

1. Stop and tell the repository owner.
2. **Rotate/revoke** any credential immediately (removal does not un-leak it).
3. For data: assess under your data-protection policy (may be reportable).
4. History rewrite (`git filter-repo`) and force-push are owner decisions;
   every collaborator must re-clone afterwards. Ask GitHub support to purge
   cached views if the repository was ever public.
