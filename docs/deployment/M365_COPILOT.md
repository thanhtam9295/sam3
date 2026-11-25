# Microsoft 365 Copilot deployment options

**Status:** templates prepared and schema-shaped, not validated in a tenant
and not deployed (T-011). Nothing is published without approval.

Three options, from least to most integration:

## Option A — Declarative agent + API plugin (recommended)

A Copilot agent whose instructions encode the analysis guidance, with one
action (`segmentImage`) calling the organisation's **internally hosted** SAM 3
service.

```
M365 Copilot ──(Entra ID OAuth)──► API gateway (APIM, validate-jwt) ──► deploy/service (GPU VM / AKS)
   declarative agent                                                        SAM 3 model, no data persisted
```

Files (`deploy/m365-copilot/`):

| File | Role |
|---|---|
| `appPackage/manifest.json` | Teams/M365 app manifest (v1.19) with `copilotAgents` |
| `appPackage/declarativeAgent.json` | agent definition (schema v1.4), conversation starters, action |
| `appPackage/instructions.txt` | agent instructions (derived from the `sam3-segmentation-analysis` skill) |
| `appPackage/ai-plugin.json` | API plugin (schema v2.2), OAuth via plugin vault |
| `appPackage/openapi.yaml` | contract of `POST /v1/segment` (kept in sync with the service by a test) |
| `appPackage/color.png`, `outline.png` | icons (192×192, 32×32) |
| `env.example` | placeholders `${{…}}` consumed by Microsoft 365 Agents Toolkit |

Steps (tenant admin involvement required):
1. Deploy `deploy/service` on approved infrastructure with a GPU:
   `uvicorn deploy.service.app:app --host 0.0.0.0 --port 8080` behind the
   gateway; set `SAM3_CHECKPOINT` to a local checkpoint path. Never set
   `SAM3_SERVICE_ALLOW_ANONYMOUS` outside local development.
2. Register an Entra ID app for the API; configure the gateway to validate
   tokens (audience, issuer, scopes) and to allow only Copilot's redirect.
3. Register the OAuth client in the Teams Developer Portal → reference id →
   `SAM3_OAUTH_REFERENCE_ID`.
4. Copy `env.example` to `deploy/m365-copilot/env/.env.dev` (git-ignored), fill values.
5. Open the folder in **Microsoft 365 Agents Toolkit** (VS Code), *Validate*
   and *Provision* to a test tenant; sideload; test the conversation starters.
6. Org-wide publishing via the admin centre only after approval and a privacy
   review (images are sent to the internal service).

Verify against current Microsoft documentation before provisioning: manifest
and schema versions evolve, and attachment hand-off from chat to API plugins is
tenant/feature dependent (KI-016; by-reference operation planned, T-010).

## Option B — Copilot Studio agent with a custom connector

Import `appPackage/openapi.yaml` as a custom connector (or REST tool) in
Copilot Studio, paste `instructions.txt` as the agent instructions and attach
`docs/ALGORITHMS.md` / the skill files as knowledge. Uses the same service and
gateway; easier for low-code owners, governed by Power Platform DLP policies.

## Option C — Agent Skills (SKILL.md) in Microsoft 365

Where the tenant's Copilot/agent platform supports the open Agent Skills
format, the skills in `.agents/skills/` (especially
`sam3-segmentation-analysis`) can be uploaded as-is: they are plain
`SKILL.md` folders with no repository-specific dependencies for the analysis
guidance. Confirm availability with the tenant administrator first; skills that
run scripts (`scientific-validation`, `sam3-testing`) are for coding agents, not
end-user Copilot.

## Security checklist

- [ ] Service reachable only through the gateway; TLS everywhere
- [ ] Entra ID token validation at the gateway; least-privilege scopes
- [ ] No image persistence or logging; request size limits (20 MB default)
- [ ] Data-classification approval for sending images to the service
- [ ] Checkpoint licence (SAM License) reviewed for the intended use
- [ ] Monitoring without payload capture; rate limits at the gateway
