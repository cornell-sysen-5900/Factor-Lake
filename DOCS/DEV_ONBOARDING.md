# Factor Lake New Member Onboarding Guide

This guide walks a new contributor through access, setup, and day-to-day workflow in the order they should do it.

## 1. Wait for access

Before you start, the professor must add you to the required workspaces:

1. [Factor-Lake GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake)
2. [Factor Lake Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm)
3. [Quant Finance Project Trello](https://trello.com/b/fFJ81SzB/quantfinanceproj)
4. [Slack Workspace](https://join.slack.com/t/sysen5900/shared_invite/zt-3wraoys1t-yQUj9Hwn7pLrFMISAqG1KQ)

Do not start work until all four are available.

## 2. Confirm you can open every tool

1. Sign in to the [GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake).
2. Open the [Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm).
3. Open the [Trello board](https://trello.com/b/fFJ81SzB/quantfinanceproj).
4. Open [Slack](https://join.slack.com/t/sysen5900/shared_invite/zt-3wraoys1t-yQUj9Hwn7pLrFMISAqG1KQ).
5. Confirm you can read the team channels and project boards.

## 3. Read the setup docs in order

Use these docs as your first local setup path:

1. Read [Documentation Contributor Guide](DOCS_CONTRIBUTOR_GUIDE.md).
2. Read [Contributing](CONTRIBUTING.md).
3. Read [Supabase Setup](SUPABASE_SETUP.md).
4. Read [Deployment](DEPLOYMENT.md) if you need to publish or verify the app.
5. Read [Factor Lake User Guide](FACTOR_LAKE_USER_GUIDE.md) to understand the app workflow.

## 4. Set up your local repository

1. Clone the [Factor-Lake GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake).
2. Create a branch before making changes.
3. Install the project dependencies.
4. Run the app or tests only after the repository is set up.

If you need the exact branch-and-commit workflow, follow [Contributing](CONTRIBUTING.md).

## 5. Understand the main tools

| Tool | What it is for | What to do first |
|---|---|---|
| [GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake) | Source code and docs | Clone it, create a branch, and read the docs before editing code |
| [Factor Lake Streamlit App](https://cornellfactorlake.streamlit.app/) | Live portfolio analysis app | Open it, load data, and verify the analysis flow |
| [Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm) | Backend data source | Use it only when data or schema changes are needed |
| [Trello board](https://trello.com/b/fFJ81SzB/quantfinanceproj) | Planning and task tracking | Check the Reference list and move cards as work changes |
| [Slack Workspace](https://join.slack.com/t/sysen5900/shared_invite/zt-3wraoys1t-yQUj9Hwn7pLrFMISAqG1KQ) | Team communication | Ask questions there instead of guessing |
| [ScrumPoker](https://www.scrumpoker-online.org/en/) | Estimation tool | Use it during sprint planning to estimate effort |
| [Streamlit Cloud](https://share.streamlit.io/) | App hosting | Use it to deploy and verify the Streamlit app |

## 6. Learn the project workflow

1. Check the [Trello board](https://trello.com/b/fFJ81SzB/quantfinanceproj) for the story you will work on.
2. Clarify the story before starting implementation.
3. Estimate the story with [ScrumPoker](https://www.scrumpoker-online.org/en/) during the team meeting.
4. Move the card to `Doing` when you start.
5. Open a pull request when your branch is ready.
6. Move the card to `Awaiting PR Approval` while it is under review.
7. After merge, the card becomes ready for demo in the professor meeting.

## 7. Use Slack correctly

1. Use the semester [Slack Workspace](https://join.slack.com/t/sysen5900/shared_invite/zt-3wraoys1t-yQUj9Hwn7pLrFMISAqG1KQ) for questions and coordination.
2. Post blockers early.
3. Ask for clarification before editing code or data.
4. Keep discussions in the semester channel so the whole team stays aligned.

## 8. Know where each deeper guide lives

1. [Documentation Contributor Guide](DOCS_CONTRIBUTOR_GUIDE.md) explains how to add or edit docs.
2. [Contributing](CONTRIBUTING.md) explains the branch-and-commit workflow.
3. [Deployment](DEPLOYMENT.md) explains how to publish the app.
4. [Supabase Setup](SUPABASE_SETUP.md) explains how to configure the database connection.
5. [Streamlit Admin Guide](STREAMLIT_ADMIN_GUIDE.md) explains production app operations.

## 9. Your first-week checklist

1. Confirm you can access all required tools.
2. Read the setup docs in Section 3.
3. Clone the repo and open the app locally.
4. Check the Trello Reference list for project links.
5. Ask one question in Slack if anything is unclear.

## 10. Quick reference

| Need | Go here |
|---|---|
| Clone and branch correctly | [Contributing](CONTRIBUTING.md) |
| Add or edit docs | [Documentation Contributor Guide](DOCS_CONTRIBUTOR_GUIDE.md) |
| Configure Supabase | [Supabase Setup](SUPABASE_SETUP.md) |
| Deploy or verify the app | [Deployment](DEPLOYMENT.md) |
| Understand the app workflow | [Factor Lake User Guide](FACTOR_LAKE_USER_GUIDE.md) |
| Ask questions | [Slack Workspace](https://join.slack.com/t/sysen5900/shared_invite/zt-3wraoys1t-yQUj9Hwn7pLrFMISAqG1KQ) |

## 11. Final rule

If something is already documented, follow the doc instead of improvising. If something is not clear, ask in [Slack](https://join.slack.com/t/sysen5900/shared_invite/zt-3wraoys1t-yQUj9Hwn7pLrFMISAqG1KQ) and update the relevant guide later if the missing step should be documented.
