# Factor-Lake

**Quantitative portfolio analysis and backtesting platform built on factor investing principles.**

Factor-Lake lets you compose multi-factor stock selection models, run historical simulations against benchmark indices, and visualise performance through an interactive Streamlit UI backed by a Supabase cloud database.

---

## Quick links

| Section | Description |
|---|---|
| [Factor Lake User Guide](FACTOR_LAKE_USER_GUIDE.md) | Student-focused walkthrough for NBA5220/Equity Research workflows |
| [Developer Onboarding](DEV_ONBOARDING.md) | Getting access, workspace setup, and understanding workflow for new team members |
| [Documentation Contributor](DOCS_CONTRIBUTOR_GUIDE.md) | How to add and edit guides, update navigation, and publish docs to the site |
| [Streamlit Admin Guide](STREAMLIT_ADMIN_GUIDE.md) | Team runbook for production deploys, secret rotation, and app operations |
| [API Reference](reference/SUMMARY.md) | Auto-generated docs for every function and class in the codebase |
| [Deployment](DEPLOYMENT.md) | How to deploy the Streamlit app |
| [Supabase Setup](SUPABASE_SETUP.md) | Database configuration and schema |
| [Supabase Maintenance](SUPABASE_MAINTENANCE_GUIDE.md) | Supabase UI operations, data hygiene, schema changes, and code mapping updates |
| [Contributing](CONTRIBUTING.md) | Development workflow and standards |
| [Security Scanning](Bandit & Safety.md) | Bandit and Safety CI integration |

---

## Codebase overview

| Package | Purpose |
|---|---|
| `src/` | Core engine — backtesting, factor registry, portfolio construction, performance metrics |
| `app/` | Streamlit UI — pages, components, config, session utilities |
| `Visualizations/` | Plotly chart builders for growth curves and portfolio composition |

---

> The **API Reference** section is generated automatically from source code at every `mkdocs build` / `mkdocs serve` run.
> No manual updates are necessary when functions are added, changed, or removed.
