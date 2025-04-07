import os
import sys
import requests

TAG = sys.argv[1]
REPO = "strangeloopcanon/LLMRank"
TOKEN = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    raise RuntimeError("GITHUB_TOKEN environment variable not set")

release_url = f"https://api.github.com/repos/{REPO}/releases"

payload = {
    "tag_name": TAG,
    "name": f"Release {TAG}",
    "body": f"Auto-published release for version {TAG}",
    "draft": False,
    "prerelease": False
}

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json"
}

r = requests.post(release_url, headers=headers, json=payload)
if r.status_code >= 300:
    print(f"GitHub release failed:\n{r.status_code}\n{r.text}")
    sys.exit(1)

print(f"✅ GitHub release {TAG} created.")
