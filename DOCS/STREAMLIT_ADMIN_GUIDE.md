# Streamlit Admin Guide (Factor Lake)

This guide is for team members who maintain the production Streamlit app. It covers common admin activities such as confirming production deploys and updating app settings.

## Scope

Use this runbook for:

- deployment verification after code merges,
- secrets management (Supabase credentials only),
- basic operational checks,
- common troubleshooting for app availability.

## Admin checklist (quick version)

1. Confirm latest code is merged to `main`.
2. Confirm Streamlit app auto-redeployed from `main`.
3. Verify basic analysis flow works.
4. Review app settings (resources, access, and limits).
5. Check docs site deploy from `main` if docs were updated.

## System overview

Production depends on three parts:

- GitHub repository (`main` branch is source of truth),
- Streamlit Community Cloud app (`app/streamlit_app.py` entrypoint),
- Supabase project (data source via `SUPABASE_URL` and `SUPABASE_KEY`).

## A. Deploy latest changes to production

### Expected behavior

Streamlit Community Cloud should redeploy automatically after a push/merge to `main`.

### Runbook

1. Confirm PR is merged into `main`.
2. Open Streamlit Community Cloud app dashboard.
3. Verify the latest commit hash in app deployment history.
4. Wait for build/restart to complete.
5. Open the live app URL.
6. Validate minimum smoke test:
   - App loads immediately (no login required),
   - `Load Market Data` succeeds,
   - `Run Portfolio Analysis` succeeds,
   - `Results` tab renders summary metrics.

### If auto-deploy does not trigger

1. In Streamlit app dashboard, use reboot/redeploy action.
2. Check build logs for dependency or import errors.
3. Confirm `pyproject.toml` includes new dependencies.
4. Re-run after fixing and merging to `main`.

## B. Update Streamlit settings (resources, access, limits)

Streamlit Community Cloud exposes settings in the app management UI. Available controls can evolve over time.

### Typical admin settings to review

- App visibility and sharing/access settings,
- restart/reboot controls,
- resource tier or limits (if available on your plan),
- environment/secrets configuration.

### "Max concurrent users" guidance

- Community Cloud does not always provide a direct, fixed "max concurrent users" knob.
- Concurrency is usually constrained by app resources and runtime behavior.
- If you need stricter concurrency control, use one or more of:
  - optimize app performance and caching,
  - split usage windows by class section,
  - move to an infrastructure option with explicit autoscaling/concurrency controls (for example container platform deployment).

## C. Secrets and configuration management

Required production secrets:

- `SUPABASE_URL`
- `SUPABASE_KEY`

Best practices:

- update secrets in Streamlit Cloud, not in committed files,
- keep `.env` for local development only,
- never commit real credentials to git,
- after secret changes, run smoke test immediately.

## D. Production smoke test (after every deploy)

1. Open app URL.
2. The app should load immediately (no login required).
3. In sidebar, keep default settings.
4. In Analysis, select at least one factor.
5. Click `Load Market Data`.
6. Click `Run Portfolio Analysis`.
7. Confirm Results sections render without errors.

## E. Common admin incidents

### Incident: data load errors

Likely causes:

- invalid `SUPABASE_URL`/`SUPABASE_KEY`,
- paused Supabase project,
- schema drift.

Actions:

1. Validate Supabase credentials in secrets.
2. Confirm Supabase project status.
3. Check app logs for failing query/field names.

### Incident: deploy succeeded but app broken

Likely causes:

- missing dependency in `pyproject.toml`,
- runtime import/path error,
- incompatible package version.

Actions:

1. Inspect Streamlit build/runtime logs.
2. Patch and merge fix to `main`.
3. Confirm new deploy hash and smoke test.

## G. Docs deploy checks (for guide updates)

MkDocs site deploys from GitHub Actions on push to `main`.

1. Open GitHub Actions and confirm `Deploy MkDocs to GitHub Pages` passed.
2. Open docs URL and verify new/updated guide appears under Guides.

## F. Semester handoff checklist

1. Validate Supabase credentials are still active.
2. Confirm app URL is reachable.
3. Run smoke test and record date/result.
4. Share updated access instructions with course staff.

## Related guides

- Factor Lake User Guide: student workflow
- Deployment: Streamlit deployment details
- Supabase Setup: data source credential setup
