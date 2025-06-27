# 🧠 Git + GitHub Collaboration Cheat Sheet

### 🎯 Real-World Scenario: Multiple Devs Working on a Calculator App

---

## 🏁 1. Project Initialization (by Lead)

```bash
git init
echo "# Calculator App" > README.md
git add .
git commit -m "Initial commit - project setup"
git branch -M main                           # Rename current branch to 'main'
git remote add origin <repo-url>            # Connect to GitHub repo (e.g., origin)
git push -u origin main                     # Push 'main' & set tracking
```

---

## 🔄 2. Daily Dev Flow (for every dev)

### ✅ Clone the Repo
```bash
git clone <repo-url>
cd calculator-app
```

### ✅ Create a Feature Branch
```bash
git checkout -b feature/addition            # Work for a new task
```

### ✅ Add & Commit Changes
```bash
git add .
git commit -m "Add addition functionality"
```

### ✅ Push to GitHub (1st time: with -u)
```bash
git push -u origin feature/addition
```

Later pushes:
```bash
git push
```

---

## 🔁 3. Pull Requests (PRs)

- Done on GitHub UI
- Compare: `feature/branch-name` → `main`
- PR Title: Short summary (e.g., "Add subtraction feature")
- Description: What changed, why, how to test

---

## 🤝 4. Code Reviews

- PR is reviewed by Lead/peers
- Comments/suggestions made
- Changes pushed to same branch (auto-updates PR)
- Once approved → Merged into `main`

---

## ⚠️ 5. Handling Merge Conflicts

Before opening PR or when PR gets stale:

```bash
git checkout feature/your-branch
git pull origin main                        # Sync with latest main
# If conflicts → resolve manually
git add .
git commit -m "Resolve merge conflict with main"
git push
```

---

## 🏷️ 6. Releasing Versions (Tagging)

After merging all features:

```bash
git checkout main
git pull origin main
git tag -a v2.0.0 -m "Release version 2.0 - full calculator"
git push origin v2.0.0
```

---

## 🔄 Git Concepts Recap

| Concept | Meaning |
|--------|---------|
| `git branch -M main` | Rename current branch to `main` |
| `git remote add origin <url>` | Connect local repo to GitHub |
| `git push -u origin <branch>` | Push and set upstream tracking |
| `git checkout -b feature/xyz` | Create & switch to a new branch |
| `git pull origin main` (on feature) | Sync latest main code into your branch |
| `git clone <repo-url>` | Clone entire repo (all branches/history) |
| PR (Pull Request) | Request to merge feature → main |
| Upstream | Default remote branch your local branch tracks |
| `origin` | Default name for the remote repo on GitHub |

---

## 🔁 Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready, stable code |
| `feature/xyz` | Work on new features or fixes |
| `release/x.y.z` | Optional: test/stage before release |
| `hotfix/issue-id` | Emergency fix to main |

---

## 🧼 Commit Hygiene

✅ Best Practices:
- One purpose per commit
- Clear messages (e.g., `"Add multiplication logic"`)
- Avoid unrelated changes in same commit
- Use present tense (`"Add"`, not `"Added"`)

---

## ⚙️ Continuous Integration (CI) Overview

- **Runs automated tests on PRs**
- Ensures no broken code is merged
- Can auto-block merging if tests fail

Example CI tool: GitHub Actions (`.github/workflows/ci.yml`)

```yaml
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: npm test  # or pytest, etc.
```

---

## 🚨 Pro Tips

- ✅ Always branch off `main`
- ✅ Pull `main` before PR to avoid conflicts
- ✅ Use PRs for **all** changes
- 🚫 Never push directly to `main`
- ✅ Review others’ code and leave feedback
- ✅ Tag releases for version tracking (`v1.0.0`, `v2.1.0`, etc.)
