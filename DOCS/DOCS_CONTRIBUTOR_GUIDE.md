# Documentation Contributor Guide

This guide is for team members who want to add or edit documentation on the Factor-Lake docs site.

## Overview

The docs site is built with [MkDocs](https://www.mkdocs.org/user-guide/writing-your-docs/) and publishes automatically to GitHub Pages when you push to `main` in the [Factor-Lake GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake). To add or edit documentation, follow this sequence:

1. Edit or create a markdown file in `DOCS/`.
2. Register the page in `mkdocs.yml`.
3. Add a quick link in `DOCS/index.md`.
4. Preview locally if needed.
5. Push to `main` and let GitHub Actions deploy the site.

## Project links

| Tool | Link |
|---|---|
| GitHub Repo | [Factor-Lake GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake) |
| Streamlit App | [Factor-Lake Streamlit App](https://cornellfactorlake.streamlit.app/) |
| Supabase Project | [Factor Lake Supabase Project](https://supabase.com/dashboard/project/ozusfgnnzanaxpcfidbm) |
| Trello Board | [Quant Finance Project Trello](https://trello.com/b/fFJ81SzB/quantfinanceproj) |
| ScrumPoker | [ScrumPoker Online](https://www.scrumpoker-online.org/en/) |
| Streamlit Cloud | [Streamlit Cloud](https://share.streamlit.io/) |

## Step 1: Create or edit your markdown file

### For a new guide

1. Create a new `.md` file in the `DOCS/` folder.
   - Example: `DOCS/MY_NEW_GUIDE.md`
2. Use a descriptive filename with no spaces.
3. Start the file with a top-level heading.
4. Write the guide as a step-by-step procedure with short sections.

Example:

```markdown
# My New Guide Title

1. Do the first thing.
2. Do the next thing.
3. Verify the result.
```

### For an existing guide

1. Open the `.md` file in `DOCS/`.
2. Rewrite the content so a new team member can follow it without guessing.
3. Replace plain-text tool names with embedded links where possible.
4. Save the file.

## Step 2: Update `mkdocs.yml`

The `mkdocs.yml` file at the repo root controls site navigation.

1. Open `mkdocs.yml`.
2. Find the `nav:` section.
3. Add your guide under `Guides` in the same style as the existing entries.
4. Make sure the display name and filename are correct and case-sensitive.

Example:

```yaml
nav:
  - Home: index.md
  - Guides:
    - Factor Lake User Guide: FACTOR_LAKE_USER_GUIDE.md
    - Developer Onboarding: DEV_ONBOARDING.md
    - New Guide Name: NEW_GUIDE.md
```

## Step 3: Update `DOCS/index.md`

The home page of the docs site has a quick-links table.

1. Open `DOCS/index.md`.
2. Find the quick-links table.
3. Add a row for the new guide.
4. Keep the link text and file name consistent with `mkdocs.yml`.

Example:

```markdown
| [New Guide Name](NEW_GUIDE.md) | Short description of what the guide covers |
```

## Step 4: Preview locally

1. Install MkDocs if you have not already:

```bash
pip install mkdocs mkdocs-material mkdocstrings mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index
```

2. From the repo root, run:

```bash
mkdocs serve
```

3. Open the local site at `http://127.0.0.1:8000/`.
4. Verify the new page appears in the navigation and renders correctly.
5. Check that links, tables, and code blocks display as expected.

## Step 5: Commit and push

1. Stage the changed files.
2. Commit with a clear message.
3. Push to `main`.

Example:

```bash
git add DOCS/MY_NEW_GUIDE.md mkdocs.yml DOCS/index.md
git commit -m "Add documentation guide"
git push origin main
```

## Step 6: Verify deployment

1. Open the [Factor-Lake GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake).
2. Check the **Actions** tab.
3. Confirm the workflow named **Deploy MkDocs to GitHub Pages** passed.
4. Open the published docs site and confirm the new page appears.

## Common pitfalls and fixes

### The page does not appear in the menu

1. Check the filename in `mkdocs.yml`.
2. Make sure the file exists in `DOCS/`.
3. Confirm the indentation in `mkdocs.yml` is correct.

### The quick link is broken

1. Check the file path in `DOCS/index.md`.
2. Use the filename only, not `DOCS/filename.md`.
3. Make sure the link text and filename match the nav entry.

### Tables or code blocks render incorrectly

1. Check Markdown syntax carefully.
2. Verify table pipes and code fences are balanced.
3. Re-run `mkdocs serve` and confirm the page builds cleanly.

### The workflow fails

1. Open the failing GitHub Actions run.
2. Read the build log for YAML or dependency errors.
3. Fix the file locally and push again.

## Markdown syntax quick reference

For detailed Markdown help, see the [MkDocs documentation](https://www.mkdocs.org/user-guide/writing-your-docs/).

### Headers

```markdown
# Level 1
## Level 2
### Level 3
```

### Lists

```markdown
- Bullet point
- Another point
  - Nested point

1. Numbered item
2. Another item
```

### Tables

```markdown
| Header 1 | Header 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |
```

### Code blocks

```markdown
```python
print("hello")
```
```
    Inline `code` with backticks

    Block code with fence:
    ```python
    def hello():
        print("Hello")
    ```
```

### Links

```markdown
[Display text](path/to/file.md)
[External link](https://example.com)
```

## Organization and naming conventions

### Filename naming

- Use UPPERCASE_WITH_UNDERSCORES for guide names (e.g., `TESTING_GUIDE.md`, `SUPABASE_MAINTENANCE_GUIDE.md`).
- Keep filenames descriptive but concise (max 40 characters).
- Avoid spaces; use underscores instead.

### Navigation ordering

Guides are organized by audience and workflow:

1. **Student Guides** (Factor Lake User Guide)
2. **Developer Guides** (Developer Onboarding, Contributing)
3. **Admin/Operations Guides** (Streamlit Admin Guide, Supabase Maintenance, Deployment)
4. **Setup and Reference** (Supabase Setup, Streamlit Styling, Security Scanning, Reorganization Summary)

When adding a new guide, place it near related guides.

### Guide templates

Consider these sections for new guides:

- **Purpose/Overview:** What problem does this guide solve?
- **Audience:** Who should read this?
- **Prerequisites:** What must the reader already know or have access to?
- **Step-by-step sections:** Clear numbered or bulleted workflows.
- **Troubleshooting:** Common issues and solutions.
- **Quick reference:** Links to related guides and resources.
- **Related guides:** Links to other relevant documentation.

## Quick checklist for adding a new guide

- [ ] Created markdown file in `DOCS/` with clear filename.
- [ ] Added entry to `mkdocs.yml` under `Guides:` with correct filename.
- [ ] Added row to quick-links table in `DOCS/index.md`.
- [ ] Tested locally with `mkdocs serve` (or manually verified syntax).
- [ ] Committed and pushed to `main` branch.
- [ ] Verified GitHub Actions workflow passed.
- [ ] Opened live docs site and confirmed guide appears and renders correctly.

## Related guides

- CONTRIBUTING.md - How to contribute code to the project
- DEPLOYMENT.md - How to deploy the Streamlit app
- Trello Reference - External tools and resources
