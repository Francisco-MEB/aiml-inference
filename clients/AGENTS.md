# Campus coding agent: working rules

These rules apply to every ndcode session, in every project.

## Verify before you say "done"
- After changing code, **run it**: execute the exact command the user will run (the script, CLI, or app entry point) and check the real output. Reading the code is not verification.
- **Exercise the feature end to end with real input.** `--help`, syntax/compile checks, and unit tests of helper functions are useful but do NOT count as verification on their own. For a new flag, run the program with that flag on a real input and inspect the full output. If it talks to a local server or API, call it for real (try it before assuming it's unavailable).
- Run the project's tests if there are any. If you added a feature, add or update a test and run it.
- If the user specified exact behaviour (an output format, a flag, "print only X"), compare the actual output against that spec character for character, and fix any mismatch.
- If something fails, fix it and re-run until it passes. Never report success on a change you haven't seen working.
- If verification needs something you can't run (a missing service, credentials), say so explicitly instead of claiming it works.

## Preserve how the code is run
- Keep existing entry points working exactly as before (e.g. `python3 path/to/script.py`, existing CLI flags, import paths) unless the user asks to change them. After a refactor, run the original command again.
- Keep changes focused on the request. Don't modify tests to make them pass unless the user asks.

## Report
- End with a short summary: what you changed, and which commands you ran to verify it, with their results.
