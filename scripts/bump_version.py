import sys
import re
from pathlib import Path

path = Path("pyproject.toml")
text = path.read_text()
match = re.search(r'version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', text)

if not match:
    raise ValueError("Version not found in pyproject.toml")

major, minor, patch = map(int, match.groups())
arg = sys.argv[1] if len(sys.argv) > 1 else "patch"

if arg == "patch":
    patch += 1
elif arg == "minor":
    minor += 1
    patch = 0
elif arg == "major":
    major += 1
    minor = patch = 0
else:
    raise ValueError("Expected patch, minor, or major")

new_version = f'{major}.{minor}.{patch}'
new_text = re.sub(
    r'version\s*=\s*"\d+\.\d+\.\d+"',
    f'version = \"{new_version}\"',
    text
)

path.write_text(new_text)
print(f"Bumped version to {new_version}")
