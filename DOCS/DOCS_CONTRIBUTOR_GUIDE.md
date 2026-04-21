# Documentation Contributor Guide

This guide is for team members who want to add new guides or edit existing documentation on the Factor Lake docs site.

## Overview

The docs site is built with MkDocs and publishes automatically to GitHub Pages when you push to `main`. To add or edit documentation, you:

1. Create or edit a markdown file in the `DOCS/` folder.
2. Update `mkdocs.yml` to register the page in navigation.
3. Update `DOCS/index.md` to add a quick link (optional but recommended).
4. Push to `main` and GitHub Actions deploys it.

## Prerequisites

- Write access to the Factor-Lake GitHub repository.
- Ability to commit and push to the `main` branch.
- Basic markdown knowledge (headers, lists, tables, links).
- Optionally: MkDocs installed locally to preview before pushing.

## Step 1: Create or edit your markdown file

### For a new guide

1. Create a new `.md` file in the `DOCS/` folder.
   - Example: `DOCS/MY_NEW_GUIDE.md`
2. Use a descriptive filename (no spaces; use underscores or hyphens).
3. Start with a top-level heading:

```markdown
# My New Guide Title

Your content here...
```

### For an existing guide

1. Open the `.md` file in the `DOCS/` folder.
2. Edit the content directly.
3. Commit and push to `main`.

## Step 2: Update mkdocs.yml navigation

The `mkdocs.yml` file at the repo root controls the site's navigation structure.

### Open mkdocs.yml

Find the `nav:` section, which looks like:

```yaml
nav:
  - Home: index.md
  - Guides:
    - Factor Lake User Guide: FACTOR_LAKE_USER_GUIDE.md
    - Developer Onboarding: DEV_ONBOARDING.md
    - Streamlit Admin Guide: STREAMLIT_ADMIN_GUIDE.md
    - Contributing: CONTRIBUTING.md
    - Deployment: DEPLOYMENT.md
    - Supabase Setup: SUPABASE_SETUP.md
    - Supabase Maintenance: SUPABASE_MAINTENANCE_GUIDE.md
    - Streamlit Styling: STREAMLIT_STYLING_GUIDE.md
    - Security Scanning: Bandit & Safety.md
    - Reorganization Summary: REORGANIZATION_SUMMARY.md
  - API Reference: reference/
```

### Add your guide to the Guides section

Under `- Guides:`, add a new line with the format:

```yaml
    - Display Name: FILENAME.md
```

**Example:** to add a new guide called `DOCS/TESTING_GUIDE.md`:

```yaml
  - Guides:
    - Factor Lake User Guide: FACTOR_LAKE_USER_GUIDE.md
    - Developer Onboarding: DEV_ONBOARDING.md
    - Testing Guide: TESTING_GUIDE.md
    - Streamlit Admin Guide: STREAMLIT_ADMIN_GUIDE.md
    ...
```

**Rules:**
- The **display name** (left side) is what users see in the navigation menu.
- The **filename** (right side) must exactly match the markdown file in DOCS/.
- Filenames are case-sensitive on GitHub.
- Organize guides in logical order (new guides often go near related ones).

## Step 3: Update DOCS/index.md (quick links)

The home page of the docs site has a quick-links table. Add your guide there so users can find it immediately.

### Open DOCS/index.md

Find the `Quick links` table:

```markdown
| Section | Description |
|---|---|
| [Factor Lake User Guide](FACTOR_LAKE_USER_GUIDE.md) | Student-focused walkthrough for NBA5220/Equity Research workflows |
| [Developer Onboarding](DEV_ONBOARDING.md) | Getting access, workspace setup, and understanding workflow for new team members |
| [API Reference](reference/SUMMARY.md) | Auto-generated docs for every function and class in the codebase |
...
```

### Add a row for your guide

Insert a new row in the table:

```markdown
| [Testing Guide](TESTING_GUIDE.md) | Best practices for writing and running tests in Factor Lake |
```

**Rules:**
- The link text should match the display name from mkdocs.yml.
- The link path should point to the markdown file.
- Write a clear, concise description (1-2 sentences).
- Keep descriptions consistent with the section's purpose.

## Step 4: Test locally (optional but recommended)

Before pushing to main, test your changes locally.

### Install MkDocs (if not already installed)

```bash
pip install mkdocs mkdocs-material mkdocstrings mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index
```

### Serve the docs locally

From the repo root:

```bash
mkdocs serve
```

You should see:

```
INFO     -  Building documentation...
INFO     -  Serving on http://127.0.0.1:8000/
```

### View your changes

1. Open http://127.0.0.1:8000/ in your browser.
2. Navigate to your guide and verify:
   - The page appears in the Guides menu.
   - The content renders correctly (tables, code blocks, links).
   - Quick links on the home page work.
3. Check for markdown syntax errors or broken links.

### Stop the server

Press Ctrl+C in the terminal.

## Step 5: Commit and push to main

Once you're satisfied with your changes:

```bash
git add DOCS/MY_NEW_GUIDE.md mkdocs.yml DOCS/index.md
git commit -m "Add: testing guide for contributor workflow"
git push origin main
```

## Step 6: Verify the deploy

### Check GitHub Actions

1. Go to GitHub: https://github.com/cornell-sysen-5900/Factor-Lake
2. Click the **Actions** tab.
3. Look for the workflow named **Deploy MkDocs to GitHub Pages**.
4. Confirm the most recent run has a ✅ (passed).
5. If it has a ❌, click the workflow to see error logs.

### Check the live site

1. Once the GitHub Actions workflow passes, the site updates automatically.
2. Open https://cornell-sysen-5900.github.io/Factor-Lake/ in your browser.
3. Navigate to your guide in the Guides menu.
4. Verify the content appears correctly.

## Common pitfalls and solutions

### "My guide doesn't appear in the menu"

**Cause:** mkdocs.yml has a typo or incorrect filename.

**Solution:**
- Check that the filename in mkdocs.yml **exactly** matches the .md file in DOCS/ (case-sensitive).
- Ensure there are no trailing spaces or special characters.
- Confirm the indentation in mkdocs.yml is correct (2 spaces per level).

### "The page renders but the quick link is broken"

**Cause:** The link path in index.md doesn't match the actual filename.

**Solution:**
- Check that the link in DOCS/index.md points to the correct filename.
- Ensure relative paths start with the filename, not a folder path (e.g., `TESTING_GUIDE.md`, not `DOCS/TESTING_GUIDE.md`).

### "Tables or code blocks don't render correctly"

**Cause:** Markdown syntax error.

**Solution:**
- Verify table pipes are properly aligned: `| Header 1 | Header 2 |`
- Ensure code blocks have proper backtick fences: `` ``` `` above and below.
- Check for unescaped special characters (e.g., `_` should be escaped as `\_` if not intended as italics).
- Test locally with `mkdocs serve` to catch issues early.

### "The GitHub Actions workflow failed"

**Cause:** mkdocs.yml syntax error or missing dependencies.

**Solution:**
- Click the failed workflow run in GitHub Actions to see the error log.
- Common errors: YAML indentation, missing quotes around filenames with special characters.
- Fix the error locally, commit, and push again.

## Markdown syntax quick reference

For more detailed markdown help, see the MkDocs documentation: https://www.mkdocs.org/user-guide/writing-your-docs/

### Headers

```markdown
# Level 1 (page title, use once)
## Level 2 (section)
### Level 3 (subsection)
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
