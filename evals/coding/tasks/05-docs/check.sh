#!/usr/bin/env bash
# Pass if harness/README.md gained a Troubleshooting section covering the three topics, and nothing else changed.
set -euo pipefail
python3 - <<'EOF'
import re, sys
text = open("harness/README.md").read()
# Drop fenced code blocks so a shell comment like "# foo" isn't mistaken for a heading.
text = re.sub(r"^```.*?^```[^\n]*$", "", text, flags=re.M | re.S)
m = re.search(r"^#+\s*Troubleshoot\w*.*$", text, re.M | re.I)
assert m, "no Troubleshooting heading"
section = text[m.end():]
nxt = re.search(r"^#{1,%d}\s" % (len(m.group(0)) - len(m.group(0).lstrip("#"))), section, re.M)
section = section[: nxt.start()] if nxt else section
topics = {
    "server not running": r"serve\.sh|not running|can't reach|cannot reach|connection",
    "model name": r"--model|model name|\bchat\b",
    "slow first answer": r"slow|first (answer|response|request)|warm|load",
}
missing = [k for k, pat in topics.items() if not re.search(pat, section, re.I)]
assert not missing, f"section doesn't cover: {missing}"
print("troubleshooting section ok,", len(section.split()), "words")
EOF
changed=$(git status --porcelain | grep -v ' harness/README.md$' || true)
[[ -z "$changed" ]] || { echo "unexpected changes:"; echo "$changed"; exit 1; }
