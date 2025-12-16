"""
Auto-Commit Script for ML Air Pollution Project
Automatically detects changes, commits, and pushes to GitHub
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)

def get_changed_files():
    """Get list of changed files"""
    code, stdout, _ = run_command("git status --porcelain")
    if code != 0:
        return []
    
    files = []
    for line in stdout.strip().split('\n'):
        if line:
            # Parse git status output
            status = line[:2]
            filename = line[3:]
            files.append((status.strip(), filename))
    return files

def generate_commit_message(changed_files):
    """Generate a descriptive commit message based on changed files"""
    if not changed_files:
        return "Update project files"
    
    # Categorize changes
    added = []
    modified = []
    deleted = []
    
    for status, filename in changed_files:
        if 'A' in status:
            added.append(filename)
        elif 'M' in status:
            modified.append(filename)
        elif 'D' in status:
            deleted.append(filename)
    
    # Build commit message
    parts = []
    if added:
        parts.append(f"Add {len(added)} file(s)")
    if modified:
        parts.append(f"Update {len(modified)} file(s)")
    if deleted:
        parts.append(f"Delete {len(deleted)} file(s)")
    
    message = " | ".join(parts) if parts else "Update project files"
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    message += f" - {timestamp}"
    
    return message

def auto_commit_and_push(custom_message=None):
    """Automatically commit and push changes to GitHub"""
    print("🔍 Checking for changes...")
    
    # Check if there are any changes
    changed_files = get_changed_files()
    
    if not changed_files:
        print("✅ No changes detected. Repository is up to date.")
        return True
    
    print(f"📝 Found {len(changed_files)} changed file(s):")
    for status, filename in changed_files[:10]:  # Show first 10
        print(f"   {status:2s} {filename}")
    if len(changed_files) > 10:
        print(f"   ... and {len(changed_files) - 10} more")
    
    # Stage all changes
    print("\n📦 Staging changes...")
    code, _, error = run_command("git add .")
    if code != 0:
        print(f"❌ Error staging files: {error}")
        return False
    
    # Generate or use custom commit message
    commit_message = custom_message or generate_commit_message(changed_files)
    print(f"\n💬 Commit message: {commit_message}")
    
    # Commit changes
    print("📝 Committing changes...")
    code, _, error = run_command(f'git commit -m "{commit_message}"')
    if code != 0:
        if "nothing to commit" in error:
            print("✅ Nothing to commit. Repository is up to date.")
            return True
        print(f"❌ Error committing: {error}")
        return False
    
    # Push to GitHub
    print("🚀 Pushing to GitHub...")
    code, stdout, error = run_command("git push origin main")
    if code != 0:
        # Try to pull first if push failed
        if "rejected" in error or "non-fast-forward" in error:
            print("⚠️  Push rejected. Pulling latest changes first...")
            code, _, pull_error = run_command("git pull origin main --rebase")
            if code != 0:
                print(f"❌ Error pulling: {pull_error}")
                return False
            
            # Try pushing again
            code, _, error = run_command("git push origin main")
            if code != 0:
                print(f"❌ Error pushing: {error}")
                return False
        else:
            print(f"❌ Error pushing: {error}")
            return False
    
    print("✅ Successfully pushed to GitHub!")
    print(f"🔗 View at: https://github.com/widgetwalker/ML_Air_Pollution")
    return True

def main():
    """Main function"""
    print("=" * 60)
    print("🌍 ML Air Pollution - Auto Commit & Push")
    print("=" * 60)
    
    # Check if we're in a git repository
    code, _, _ = run_command("git rev-parse --git-dir")
    if code != 0:
        print("❌ Not a git repository. Please run 'git init' first.")
        sys.exit(1)
    
    # Get custom message from command line if provided
    custom_message = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    
    # Run auto-commit
    success = auto_commit_and_push(custom_message)
    
    print("=" * 60)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
