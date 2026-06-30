# RAG v0 - Configuration Baseline Correction

This file records a V0 documentation correction.

The original pre-V1 `config.py` source could not be recovered from Git history:

- `backend/config.py` was first tracked in the V1 commit.
- no tracked root-level `config.py` exists in the pre-V1 commit.

Known V0 configuration state (pre-V1 — corrupted in this repo, not recoverable from Git history):

- `config.py` used `SettingsConfigDict(fields={...})` — invalid syntax for `pydantic-settings` v2 (treated as a no-op / ignored).
- `bearer_token` and `gemini_api_key` were not required (or had masking defaults), so missing keys did not fail fast at startup.
- A leftover `voyage_api_key` field existed even though `voyageai` is banned by project rules.


The current, fixed V1 configuration is documented in `docs/RAG_v1.md`.
