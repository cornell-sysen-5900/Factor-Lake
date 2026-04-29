## How to contribute to Factor Lake

Use this workflow whenever you make a change in the [Factor-Lake GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake).

### Step 1: Clone the repository

```bash
git clone https://github.com/cornell-sysen-5900/Factor-Lake.git
```

### Step 2: Move into the project folder

```bash
cd Factor-Lake
```

### Step 3: Check your branch state

```bash
git status
```

### Step 4: Create a feature branch

```bash
git checkout -b <new-branch>
```

### Step 5: Confirm the branch changed

```bash
git branch
```

The active branch will have an asterisk.

### Step 6: Edit the files you need

Use VS Code or your preferred editor to make the change.

### Step 7: Stage the files

```bash
git add <filename>
```

### Step 8: Commit the change

```bash
git commit -m "Describe the change clearly"
```

### Step 9: Push the branch

```bash
git push origin <new-branch>
```

### Step 10: Open a pull request

1. Go to the [Factor-Lake GitHub Repo](https://github.com/cornell-sysen-5900/Factor-Lake).
2. Open the branch comparison view.
3. Create a pull request into `main`.
4. Describe what changed and how you tested it.

### Diagram System Breakdown
<details markdown="1">
<summary>Portfolio Use Case Diagram</summary>

  ![Portfolio Use Case Diagram](./GoogleColabDiagrams/UseCaseDiagram.png)
  <sub><i>This diagram explains how the developer and the portfolio manager interact with the system. The portfolio manager(s) primarily interact with Google Drive and Google Colab, while the developer manages both code and computation via Google Colab and GitHub. This diagram aids in identifying access points and roles therefore supporting secure coding and permission management. </i></sub>
  


</details>

<details markdown="1">
<summary>Class Diagram</summary>

  ![Class Diagram](./GoogleColabDiagrams/ClassDiagram.png)
  <sub><i>This diagram shows the structure and relationships between major classes in the portfolio construction system. It highlights how user inputs, market data, and various factor classes interact to compute portfolio holdings, returns, and analytics. It is useful for developers that are planning for feature extensions, testing coverage, and/or debugging.</i></sub>

</details>

<details markdown="1">
<summary>Deployment Diagram</summary>

  ![Deployment Diagram](./GoogleColabDiagrams/DeploymentDiagram.png)
  <sub><i>This deployment diagram outlines how a user accesses Google Colab to run the factor portfolio notebook. It includes the OAuth-based authentication flow via Cornell IDP and Duo, Google Drive integration for data storage, and repository access from GitHub. It maps the data flow and token exchanges required to mount storage and retrieve market data securely. This is helpful for onboarding new collaborators, security reviews, and cloud resource planning. </i></sub>

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
