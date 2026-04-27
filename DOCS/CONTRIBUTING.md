## How to contribute to Factor Lake

### Git Basics

The Quebec team follows the standard git workflow to manage this repository. Beginners will find the steps detailed below helpful in making their first contributions.

1. Clone Repository

```
git clone https://github.com/cornell-sysen-5900/Factor-Lake.git
```

2. Move to repository directory 

```
cd Factor-Lake
```

3. Check status and confirm its up to date.

```
git status
```

4. Create a new branch.

```
git checkout -b <new branch>
```

5. Check that you're working in the newly created branch.

```
git branch
```

The branch your working on will be marked with an asterisk.

6. Add a new file or make changes to an existing file using VSCode or preferred editor.

7. Stage file for next commit.

```
git add <filename>
```
8. Commit changes with descriptive message.

```
git commit -m "Added new file and this is my descriptive message."
```

9. Push changes to repository branch.

```
git push origin <branch you created in step 4>
```
### Diagram System Breakdown
<details markdown="1">
<summary>Portfolio Use Case Diagram</summary>

  ![Portfolio Use Case Diagram](./GoogleColabDiagrams/UseCaseDiagram.png)
  <sub><i>This diagram explains how the portfolio manager and developer interact with the current Streamlit product. It focuses on the real UI flow in the repository today: configure inputs, load Supabase data, run the backtest, review results, and maintain the code, docs, and test workflow through GitHub.</i></sub>
  


</details>

<details markdown="1">
<summary>Class Diagram</summary>

  ![Class Diagram](./GoogleColabDiagrams/ClassDiagram.png)
  <sub><i>This diagram maps the active module and class relationships in the current codebase. It shows how the Streamlit UI, session-state helpers, Supabase ingestion, backtest engine, benchmark data, performance metrics, visualizations, and the remaining core classes work together during analysis.</i></sub>

</details>

<details markdown="1">
<summary>Deployment Diagram</summary>

  ![Deployment Diagram](./GoogleColabDiagrams/DeploymentDiagram.png)
  <sub><i>This deployment diagram reflects the present Streamlit-oriented setup. It shows the browser client, the Streamlit runtime, the Supabase cloud database, local benchmark and metrics modules, and the GitHub-backed code and documentation workflow instead of the older Google Colab and Google Drive path.</i></sub>

</details>

### SecDevOps Approach
#### General Approach
The Factor Lake team has agreed to take a SecDevOps approach to our project's development and all contributors are expected to do the same. SecDevOps is a software development approach that prioritizes security.[^1] All contributors must abide by Factor Lake's Secure Coding Standard.

#### Static Analysis Using Bandit

Static analysis techniques are used to evaluate a source code's formatting consistency, adherence to coding standards, documentation conventions, and security vulnerabilities **without** executing it. Per Snyk.io, "static code analysis will enable us to detect code bugs or vulnerabilities that other testing methods and tools, such as manual code reviews and compilers, frequently miss." [^2] 

Static Application Security Testing (SAST) is a subset of static analysis testing that our team has decided to focus on. SAST prioritizes the detection of security vulnerabilities as opposed to things like code style deviations and optimization issues. SAST tools are "designed specifically to find security issues with high accuracy, striving for low false positive and false negative rates, and providing detailed information about root causes and remedies of spotted vulnerabilities."[^3] 

The first tool we've decided to use for SAST is [Bandit](https://bandit.readthedocs.io/en/latest/). Per their documentation, "Bandit is a tool designed to find common security issues in Python code. To do this, Bandit processes each file, builds an Abstract Syntax Tree (AST) from it, and runs appropriate plugins against the AST nodes. Once Bandit has finished scanning all the files, it generates a report."[^4]

The second tool we've decided to use for SAST is Safety CLI. Per their documentation, "Safety CLI is a Python dependency vulnerability scanner designed to enhance software supply chain security and enable the secure use of Python packages."[^5] In order for Safety CLI to work, our repository's dependencies must be stored in a requirements.txt file.

[^1]: https://www.pluralsight.com/blog/software-development/secdevops#:~:text=with%20Pluralsight%20Flow-,What%20is%20SecDevOps?,vulnerabilities%20they%20missed%20earlier%20on
[^2]: https://snyk.io/learn/open-source-static-code-analysis/
[^3]: https://snyk.io/learn/open-source-static-code-analysis/
[^4]: https://bandit.readthedocs.io/en/latest/index.html
[^5]: https://safetycli.com
