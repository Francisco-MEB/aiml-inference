#!/usr/bin/env bash
# Plant a bug in harness/chat.py's stats line and add a unit test that exposes it.
set -euo pipefail
sed -i "s/{'~' if s\['approx'\] else ''}/{'~' if not s['approx'] else ''}/" harness/chat.py
grep -q "if not s\['approx'\]" harness/chat.py || { echo "setup: failed to plant bug" >&2; exit 1; }
mkdir -p tests
cat > tests/test_chat_stats.py <<'EOF'
import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location(
    "chat", pathlib.Path(__file__).resolve().parent.parent / "harness" / "chat.py")
chat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat)
chat.TTY = False  # no ANSI colour codes in the formatted string


class FormatStatsTest(unittest.TestCase):
    def test_exact_token_count_has_no_tilde(self):
        s = {"ttft": 0.5, "tokens": 100, "approx": False, "tok_s": 20.0, "total": 5.0}
        self.assertEqual(chat.format_stats(s), "[TTFT 0.50s | 100 tokens | 20.0 tok/s | total 5.0s]")

    def test_estimated_token_count_has_tilde(self):
        s = {"ttft": None, "tokens": 7, "approx": True, "tok_s": None, "total": 1.0}
        self.assertEqual(chat.format_stats(s), "[~7 tokens | total 1.0s]")


if __name__ == "__main__":
    unittest.main()
EOF
