# Factor-Lake

**Quantitative portfolio analysis and backtesting platform built on factor investing principles.**

Factor-Lake lets you compose multi-factor stock selection models, run historical simulations against benchmark indices, and visualise performance through an interactive Streamlit UI backed by a Supabase cloud database.

---

## Quick links

| Section | Description |
|---|---|
| [API Reference](reference/) | Auto-generated docs for every function and class in the codebase |
| [Deployment](DEPLOYMENT.md) | How to deploy the Streamlit app |
| [Supabase Setup](SUPABASE_SETUP.md) | Database configuration and schema |
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

> The **API Reference** section is generated automatically from source code at every build.
> No manual updates are necessary when functions are added, changed, or removed.
