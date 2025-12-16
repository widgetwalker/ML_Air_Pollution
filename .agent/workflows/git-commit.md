---
description: Quick commit and push changes to GitHub
---

# Git Commit and Push Workflow

This workflow helps you quickly commit and push changes to the GitHub repository.

## Quick Commit with Auto-Generated Message

```bash
// turbo
python auto_commit.py
```

This will:
- Detect all changed files
- Generate a descriptive commit message
- Commit the changes
- Push to GitHub

## Commit with Custom Message

```bash
python auto_commit.py "Your custom commit message here"
```

## Manual Git Operations

### Check Status

```bash
git status
```

### Add Specific Files

```bash
git add <filename>
```

### Commit Manually

```bash
git commit -m "Your commit message"
```

### Push to GitHub

```bash
git push origin main
```

### Pull Latest Changes

```bash
git pull origin main
```

### View Commit History

```bash
git log --oneline -10
```

## Notes

- The auto-commit script automatically handles conflicts by pulling and rebasing
- `.env` file is excluded from commits (contains API keys)
- Large model files may take time to push on first commit
