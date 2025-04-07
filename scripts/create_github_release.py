import os
import sys
import requests

TAG = sys.argv[1]
REPO = "strangeloopcanon/LLMRank"
TOKEN = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    raise RuntimeError("GITHUB_TOKEN environment variable not set")

BASE_URL = f"https://api.github.com/repos/{REPO}"
RELEASE_URL = f"{BASE_URL}/releases"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json"
}

# Check if release already exists
r = requests.get(f"{BASE_URL}/releases/tags/{TAG}", headers=HEADERS)
if r.status_code == 200:
    print(f"⚠️ GitHub release for tag {TAG} already exists. Skipping.")
    sys.exit(0)
elif r.status_code != 404:
    print(f"GitHub release check failed:\n{r.status_code}\n{r.text}")
    sys.exit(1)

# Create release
payload = {
    "tag_name": TAG,
    "name": f"Release {TAG}",
    "body": f"Auto-published release for version {TAG}",
    "draft": False,
    "prerelease": False
}

r = requests.post(RELEASE_URL, headers=HEADERS, json=payload)
if r.status_code >= 300:
    print(f"GitHub release creation failed:\n{r.status_code}\n{r.text}")
    sys.exit(1)

print(f"✅ GitHub release {TAG} created.")
