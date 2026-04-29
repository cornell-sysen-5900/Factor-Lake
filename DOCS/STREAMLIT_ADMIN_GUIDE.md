# Streamlit Admin Workflow

Use this guide when you need to keep the production [Factor-Lake Streamlit App](https://cornellfactorlake.streamlit.app/) running correctly.

## 1. Know the production path

1. Source of truth lives in the [Factor-Lake GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake).
2. Production app runs from `app/streamlit_app.py`.
3. Data comes from the [Factor Lake Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm).
4. Deploys happen through [Streamlit Cloud](https://share.streamlit.io/).

## 2. Check the app after every merge

1. Confirm the change is merged to `main`.
2. Open the Streamlit Cloud app dashboard.
3. Confirm the latest commit is the one you expect.
4. Wait for the build or restart to finish.
5. Open the live app URL.

## 3. Run the production smoke test

1. Leave the default sidebar settings in place.
2. Select at least one factor in Analysis.
3. Click Load Market Data.
4. Click Run Portfolio Analysis.
5. Confirm the Results tab renders metrics and charts.

## 4. If the app did not redeploy

1. Open the app logs in Streamlit Cloud.
2. Check for import, dependency, or secret errors.
3. Confirm `requirements.txt` is current.
4. Trigger a restart or redeploy if needed.
5. Re-test the app after the fix.

## 5. Manage secrets safely

1. Keep `SUPABASE_URL` and `SUPABASE_KEY` in Streamlit Cloud secrets for production.
2. Keep `.env` local for development.
3. Never commit real secrets.
4. After any secret change, rerun the smoke test.

## 6. Review app settings when needed

1. Open the Streamlit Cloud app settings.
2. Review app visibility and access.
3. Review restart and resource settings.
4. Review any deployment-related configuration.
5. Use caching or usage windows if you need to reduce pressure on the app.

## 7. Handle common incidents

### Data load error

1. Check the app logs.
2. Check the Supabase project status.
3. Confirm the secrets are correct.
4. Confirm the table and required columns still exist.

### App loads but analysis fails

1. Check for missing dependencies or import errors.
2. Check whether the schema changed.
3. Confirm the factor mappings still match the data.
4. Re-test after the fix.

## 8. Check docs deploys when docs change

1. Open GitHub Actions in the [Factor-Lake GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake).
2. Confirm `Deploy MkDocs to GitHub Pages` passed.
3. Open the published docs site.
4. Confirm the new guide appears under Guides.

## 9. Finish a semester handoff

1. Confirm app access still works.
2. Confirm Supabase credentials are still active.
3. Run and record a smoke test.
4. Share the current access and support path with the course staff.

## 10. Related guides

1. [Factor Lake User Guide](FACTOR_LAKE_USER_GUIDE.md)
2. [Deployment](DEPLOYMENT.md)
3. [Supabase Setup](SUPABASE_SETUP.md)

## 11. Reference

Use this section when you want the operational details that sit behind the checklist.

1. The app entrypoint is `app/streamlit_app.py`.
2. The production app reads data from the [Factor Lake Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm).
3. Streamlit Cloud handles the deploy after pushes to `main`.
4. `SUPABASE_URL` and `SUPABASE_KEY` must exist in production secrets.
5. The smoke test should always include data load, analysis run, and Results rendering.
6. If docs changed, verify the GitHub Actions workflow for MkDocs as part of the same release check.
