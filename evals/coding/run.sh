#!/usr/bin/env bash
# Coding-agent acceptance eval: run each task with each agent ("arm") in a
# throwaway clone of this repo, then check the result automatically.
#
#   evals/coding/run.sh                                   # all tasks x all arms
#   evals/coding/run.sh --arms ndcode --tasks 02-fix-bug  # a subset
#
# Arms:
#   ndcode        OpenCode + campus vLLM            (what campus users get)
#   claude-local  Claude Code -> campus vLLM        (same model, Anthropic's agent, via /v1/messages)
#   claude-cloud  Claude Code -> Anthropic's cloud  (quality ceiling; uses your Claude account)
#
# Agents run with permission prompts auto-approved, which is safe ONLY because
# each run happens in a disposable clone under $OUT. Never point this at a real checkout.
#
# Output: $OUT/results.csv plus per-run logs. Needs the campus vLLM server running.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ARMS="ndcode,claude-local,claude-cloud"
TASKS="all"
OUT="${OUT:-${TMPDIR:-/tmp}/coding-eval-$(date +%Y%m%d-%H%M%S)}"
VLLM="${VLLM_URL:-http://127.0.0.1:8000}"
TIMEOUT="${TIMEOUT:-1200}"   # seconds per run

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arms) ARMS="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

curl -sf -m 5 "$VLLM/v1/models" >/dev/null || { echo "vLLM isn't reachable at $VLLM. Start serving/vllm/serve.sh first." >&2; exit 1; }
mkdir -p "$OUT"
CSV="$OUT/results.csv"
echo "task,arm,pass,seconds,gen_tokens,prompt_tokens,cost_usd" > "$CSV"

# Sum a vLLM Prometheus counter across label sets (e.g. vllm:generation_tokens_total).
metric() { curl -s -m 5 "$VLLM/metrics" | awk -v n="$1" 'index($1, n) == 1 && $1 !~ /_created/ {s += $NF} END {printf "%d", s}'; }

# Claude Code against our server: every model alias maps to the campus model.
claude_local_env=(
  ANTHROPIC_BASE_URL="$VLLM"
  ANTHROPIC_AUTH_TOKEN=campus
  ANTHROPIC_MODEL=chat
  ANTHROPIC_DEFAULT_OPUS_MODEL=chat
  ANTHROPIC_DEFAULT_SONNET_MODEL=chat
  ANTHROPIC_DEFAULT_HAIKU_MODEL=chat
  ANTHROPIC_SMALL_FAST_MODEL=chat
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
)

if [[ "$TASKS" == all ]]; then
  task_dirs=("$HERE"/tasks/*/)
else
  task_dirs=(); IFS=, read -ra ids <<<"$TASKS"
  for id in "${ids[@]}"; do task_dirs+=("$HERE/tasks/$id/"); done
fi
IFS=, read -ra arms <<<"$ARMS"

for tdir in "${task_dirs[@]}"; do
  tdir="${tdir%/}"; task="$(basename "$tdir")"
  [[ -f "$tdir/prompt.md" ]] || { echo "no such task: $task" >&2; continue; }
  prompt="$(cat "$tdir/prompt.md")"
  for arm in "${arms[@]}"; do
    run="$OUT/$task.$arm"; work="$run/repo"
    mkdir -p "$run"
    git clone -q "$REPO" "$work"
    if [[ -f "$tdir/setup.sh" ]]; then
      (cd "$work" && bash "$tdir/setup.sh" && git add -A && git -c user.name=eval -c user.email=eval@localhost commit -qm "eval setup") \
        || { echo "$task,$arm,setup-failed,,,," >> "$CSV"; continue; }
    fi

    echo "== $task / $arm"
    g0=$(metric vllm:generation_tokens_total); p0=$(metric vllm:prompt_tokens_total)
    t0=$(date +%s.%N)
    case "$arm" in
      ndcode)
        (cd "$work" && timeout "$TIMEOUT" ndcode run --auto "$prompt") > "$run/agent.log" 2>&1 ;;
      claude-local)
        (cd "$work" && env "${claude_local_env[@]}" timeout "$TIMEOUT" claude -p "$prompt" \
          --dangerously-skip-permissions --output-format json) > "$run/agent.log" 2>&1 ;;
      claude-cloud)
        (cd "$work" && timeout "$TIMEOUT" claude -p "$prompt" \
          --dangerously-skip-permissions --output-format json) > "$run/agent.log" 2>&1 ;;
      *) echo "unknown arm $arm" >&2; continue ;;
    esac
    agent_rc=$?
    t1=$(date +%s.%N)
    g1=$(metric vllm:generation_tokens_total); p1=$(metric vllm:prompt_tokens_total)

    if (cd "$work" && EVAL_BASE_URL="$VLLM/v1" bash "$tdir/check.sh") > "$run/check.log" 2>&1; then pass=1; else pass=0; fi
    secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.0f", b - a}')
    cost=""
    if [[ "$arm" == claude-* ]]; then
      cost=$(python3 -c 'import json,sys
try: print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1]).get("total_cost_usd", ""))
except Exception: print("")' "$run/agent.log")
    fi
    gen=$((g1 - g0)); prm=$((p1 - p0))
    [[ "$arm" == claude-cloud ]] && gen="" && prm=""   # cloud tokens don't hit our server
    echo "   pass=$pass time=${secs}s agent_exit=$agent_rc gen_tokens=$gen prompt_tokens=$prm cost=$cost"
    echo "$task,$arm,$pass,$secs,$gen,$prm,$cost" >> "$CSV"
  done
done

echo; echo "Results: $CSV"; column -s, -t < "$CSV"
