#!/usr/bin/env python3
"""Regenerates prompts_v1.jsonl. The committed .jsonl (and its sha256 in the config) is the
source of truth; this script only documents where the text came from. All content is synthetic."""
import json
import pathlib
import random

rng = random.Random(20261115)
DEPTS = ["Applied Math", "Biology", "Chemistry", "Civil Eng", "Computer Science", "Economics",
         "History", "Linguistics", "Mech Eng", "Philosophy", "Physics", "Statistics"]
TOPICS = ["Foundations", "Methods", "Systems", "Modeling", "Seminar", "Laboratory", "Theory",
          "Design", "Analysis", "Practicum"]
ROOMS = ["Hall A", "Hall B", "Annex 2", "Lab 14", "Room 210", "Room 318", "Studio 5"]
DAYS = ["Mon/Wed", "Tue/Thu", "Mon/Wed/Fri", "Fri"]

rows = []
for i in range(1, 49):
    dept = rng.choice(DEPTS)
    rows.append(
        f"Record {i:03d}: course {dept[:3].upper()}-{rng.randrange(100, 500)} "
        f"\"{rng.choice(TOPICS)} of {dept} {rng.choice(['I', 'II', 'III'])}\", "
        f"{rng.choice([1, 3, 3, 4])} credits, meets {rng.choice(DAYS)} at "
        f"{rng.randrange(8, 18)}:{rng.choice(['00', '30'])} in {rng.choice(ROOMS)}, "
        f"capacity {rng.randrange(12, 120)}, enrolled {rng.randrange(5, 110)}, "
        f"instructor ID {rng.randrange(1000, 9999)}."
    )
catalog = "\n".join(rows)

prompts = [
    {"id": "short_chat", "class": "short_chat",
     "messages": [{"role": "user", "content":
                   "Explain in plain language what a hash table is and why lookups are fast."}]},
    {"id": "long_prefill", "class": "long_context_prefill",
     "messages": [{"role": "user", "content":
                   "Below is a synthetic course catalog extract. Read all of it, then answer the "
                   "question at the end.\n\n" + catalog + "\n\nQuestion: Summarize which "
                   "departments appear most often, point out any records where enrollment "
                   "exceeds capacity, and describe the scheduling patterns you notice."}]},
    {"id": "coding", "class": "coding",
     "messages": [{"role": "user", "content":
                   "Write a Python function `merge_intervals(intervals)` that takes a list of "
                   "[start, end] pairs, merges all overlapping intervals and returns the result "
                   "sorted by start. Include type hints, a docstring, handling for an empty "
                   "list, and three assert-based tests. Then explain the time complexity."}]},
]
out = pathlib.Path(__file__).with_name("prompts_v1.jsonl")
out.write_text("".join(json.dumps(p, ensure_ascii=True) + "\n" for p in prompts))
print(out, sum(len(json.dumps(p)) for p in prompts), "chars")
