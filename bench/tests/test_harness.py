"""Tests for the aiml-bench/2.0 harness. Standard library unittest; no GPU, no network
(except where noted), no real server: engines can be swapped for tests/fake_server.py.

  .venv/bin/python -m unittest discover -s bench/tests -v
"""
import copy
import csv
import hashlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH / "harness"))

import catalog  # noqa: E402
import common  # noqa: E402
import engines  # noqa: E402
import hf  # noqa: E402
import protocol  # noqa: E402
import run_quality  # noqa: E402

FAKE = str(HERE / "fake_server.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def plan_cfg(name="qwen", machine="dell"):
    p = catalog.plan_study(name, catalog.parse_machine(machine))
    return catalog.study_config(p, "tester", today="20990101")


class CatalogTest(unittest.TestCase):
    def test_every_build_is_pinned_to_a_commit(self):
        cat = catalog.load()
        for fam in cat["families"].values():
            for builds in fam["builds"].values():
                for b in builds:
                    for mk, x in b["models"].items():
                        self.assertRegex(x.get("revision", ""), r"^[0-9a-f]{40}$", x["repo"])
                        self.assertGreater(x.get("weights_gb") or 0, 0, x["repo"])
                        self.assertIn(mk, fam["models"])

    def test_dell_qwen_is_nvfp4_small_to_large(self):
        p = catalog.plan_study("qwen", catalog.parse_machine("dell"))
        self.assertEqual(p["build"], "nvfp4")
        self.assertEqual([m["key"] for m in p["models"]],
                         ["qwen3-8b", "qwen3.8-27b", "qwen3.6-35b-a3b", "qwen3.5-122b-a10b"])
        self.assertTrue(all(m["quantization"] == "nvfp4" for m in p["models"]))

    def test_dell_gemma_fits_at_bf16(self):
        self.assertEqual(catalog.plan_study("gemma", catalog.parse_machine("dell"))["build"],
                         "bf16")

    def test_small_mac_drops_what_does_not_fit_and_raises_precision(self):
        p = catalog.plan_study("qwen", catalog.parse_machine("mac:64"))
        self.assertEqual(p["build"], "8bit")
        keys = [m["key"] for m in p["models"]]
        self.assertNotIn("qwen3.5-122b-a10b", keys)
        self.assertEqual([x["key"] for x in p["left_out"]], ["qwen3.5-122b-a10b"])

    def test_model_without_a_build_in_the_study_format_is_reported(self):
        p = catalog.plan_study("mistral", catalog.parse_machine("dell"))
        self.assertIn("ministral3-14b", [x["key"] for x in p["left_out"]])

    def test_single_model_and_repo_lookup(self):
        p = catalog.plan_study("gpt-oss-120b", catalog.parse_machine("dell"))
        self.assertEqual([m["key"] for m in p["models"]], ["gpt-oss-120b"])
        self.assertEqual(catalog.find(catalog.load(), "openai/gpt-oss-20b"),
                         ("gpt-oss", "gpt-oss-20b"))

    def test_single_model_uses_its_family_study_build(self):
        # Qwen3 8B alone would fit at FP8, but the Qwen study on a GB10 is NVFP4.
        p = catalog.plan_study("qwen3-8b", catalog.parse_machine("dell"))
        self.assertEqual(p["build"], "nvfp4")
        self.assertEqual([m["repo"] for m in p["models"]], ["nvidia/Qwen3-8B-NVFP4"])

    def test_unknown_model_is_refused(self):
        with self.assertRaises(SystemExit):
            catalog.plan_study("not-a-model", catalog.parse_machine("dell"))

    def test_multimodal_checkpoints_serve_text_only_on_vllm(self):
        p = catalog.plan_study("gemma", catalog.parse_machine("dell"))
        self.assertTrue(all("--language-model-only" in m["server_args"] for m in p["models"]))


class ProtocolTest(unittest.TestCase):
    def test_planned_config_is_compliant(self):
        self.assertEqual(protocol.check_config(plan_cfg()), [])

    def test_seed_mixed_quant_and_branch_revisions_are_flagged(self):
        cfg = plan_cfg()
        cfg["request"]["seed"] = 42
        cfg["models"][1]["quantization"] = "fp8"
        cfg["models"][0]["revision"] = "main"
        bad = " | ".join(protocol.check_config(cfg))
        self.assertIn("seed", bad)
        self.assertIn("differs from the study's", bad)
        self.assertIn("commit SHA", bad)

    def test_server_args(self):
        v = dict(protocol.server_args("vllm", 8, chat_template_kwargs={"enable_thinking": False}))
        self.assertEqual(v["--max-num-seqs"], "8")
        self.assertEqual(v["--max-model-len"], "4096")
        self.assertEqual(v["--gpu-memory-utilization"], "0.85")
        self.assertEqual(json.loads(v["--default-chat-template-kwargs"]), {"enable_thinking": False})
        m = dict(protocol.server_args("mlx", 8))
        self.assertEqual((m["--decode-concurrency"], m["--prompt-concurrency"]), ("8", "8"))

    def test_slots_of_new_and_legacy_runs(self):
        self.assertEqual(common.slots_of({"slots": 16}), "16")
        self.assertEqual(common.slots_of({"server_env_set_by_harness":
                                          {"OLLAMA_NUM_PARALLEL": "4"}}), "4")
        self.assertEqual(common.slots_of({}), "")


class HfTest(unittest.TestCase):
    def test_mistral_duplicate_weights_count_once(self):
        info = {"siblings": [
            {"rfilename": "consolidated.safetensors", "lfs": {"sha256": "a", "size": 10}},
            {"rfilename": "model-00001-of-00001.safetensors", "lfs": {"sha256": "b", "size": 10}},
            {"rfilename": "original/model.safetensors", "lfs": {"sha256": "c", "size": 10}}]}
        self.assertEqual(list(hf.weight_files(info)), ["model-00001-of-00001.safetensors"])
        self.assertIn("consolidated*", hf.download_excludes(info))

    def test_verify_cache(self):
        with tempfile.TemporaryDirectory() as root:
            data = b"weights" * 1000
            sha = hashlib.sha256(data).hexdigest()
            repo, rev = "org/model", "0" * 40
            base = pathlib.Path(root) / "models--org--model"
            (base / "blobs").mkdir(parents=True)
            (base / "blobs" / sha).write_bytes(data)
            snap = base / "snapshots" / rev
            snap.mkdir(parents=True)
            os.symlink(base / "blobs" / sha, snap / "model.safetensors")
            files = {"model.safetensors": {"sha256": sha, "size": len(data)}}
            self.assertTrue(hf.verify_cache(repo, rev, files, root, rehash=True)["ok"])
            wrong = {"model.safetensors": {"sha256": sha, "size": len(data) + 1}}
            self.assertFalse(hf.verify_cache(repo, rev, wrong, root)["ok"])
            (base / "blobs" / sha).write_bytes(b"tampered" * 875)     # same size, other bytes
            self.assertTrue(hf.verify_cache(repo, rev, files, root, rehash=False)["ok"])
            self.assertFalse(hf.verify_cache(repo, rev, files, root, rehash=True)["ok"])
            missing = {"other.safetensors": {"sha256": sha, "size": 1}}
            self.assertEqual(hf.verify_cache(repo, rev, missing, root)["missing"],
                             ["other.safetensors"])

    def test_verify_cache_through_a_shared_blob_store(self):
        # huggingface_hub 2.x: blobs/<sha256> is itself a link into a shared store.
        with tempfile.TemporaryDirectory() as root:
            data = b"weights" * 1000
            sha = hashlib.sha256(data).hexdigest()
            repo, rev = "org/model", "1" * 40
            store = pathlib.Path(root) / "blobs" / "ab"
            store.mkdir(parents=True)
            (store / "abcdef-other-hash").write_bytes(data)
            base = pathlib.Path(root) / "models--org--model"
            (base / "blobs").mkdir(parents=True)
            os.symlink(store / "abcdef-other-hash", base / "blobs" / sha)
            snap = base / "snapshots" / rev
            snap.mkdir(parents=True)
            os.symlink(pathlib.Path("../../blobs") / sha, snap / "model.safetensors")
            files = {"model.safetensors": {"sha256": sha, "size": len(data)}}
            self.assertTrue(hf.verify_cache(repo, rev, files, root, rehash=True)["ok"])

    def test_parse_vllm_metrics(self):
        text = ('vllm:cache_config_info{block_size="16",cache_dtype="fp8",'
                'enable_prefix_caching="True",num_gpu_blocks="1097"} 1.0\n'
                'vllm:num_preemptions_total{engine="0",model_name="m"} 3.0\n')
        m = engines.parse_vllm_metrics(text)
        self.assertEqual(m["kv_cache_tokens"], 1097 * 16)
        self.assertEqual(m["kv_cache_dtype"], "fp8")
        self.assertEqual(m["preemptions"], 3.0)


class QualityTest(unittest.TestCase):
    def test_limits_split_into_calls(self):
        cfg = plan_cfg()
        groups = dict((lim, ts) for ts, lim in run_quality.task_plan(cfg))
        self.assertEqual(groups[100], ["mmlu_pro"])
        self.assertIn("aime25", groups[None])
        self.assertEqual(run_quality.task_plan(cfg, 5), [(cfg["quality"]["tasks"], 5)])

    def test_gen_kwargs_carry_the_model_thinking_setting(self):
        cfg = plan_cfg("gpt-oss")
        kw = run_quality.gen_kwargs(cfg, cfg["models"][0])
        self.assertIn("temperature=0", kw)
        self.assertIn("reasoning_effort=low", kw)

    def test_lm_eval_payload_has_no_seed(self):
        try:
            import lm_eval_bench
        except ImportError:
            self.skipTest("lm-eval not installed in this interpreter")
        obj = lm_eval_bench.BenchChatCompletion.__new__(lm_eval_bench.BenchChatCompletion)
        obj.model, obj._max_gen_toks = "m", 16
        payload = obj._create_payload([{"role": "user", "content": "hi"}],
                                      gen_kwargs={"temperature": 0, "until": ["\n"]})
        self.assertNotIn("seed", payload)
        self.assertEqual(payload["temperature"], 0)


class FakeServerTest(unittest.TestCase):
    def test_server_start_stream_and_stop(self):
        port = free_port()
        cfg = plan_cfg()
        cfg["backend"]["port"] = port
        m = cfg["models"][0]
        old = os.environ.get("BENCH_FAKE_SERVER")
        os.environ["BENCH_FAKE_SERVER"] = FAKE
        try:
            with tempfile.TemporaryDirectory() as d:
                srv = engines.Server(cfg, m, 2, pathlib.Path(d) / "server.log")
                with srv:
                    info = srv.info()
                    self.assertEqual(info["engine_version"], "0.0-fake")
                    self.assertEqual(info["kv_cache_tokens"], 16000)
                    self.assertEqual(info["weights_loaded_gib"], 1.0)
                    req = dict(cfg["request"], max_tokens=8, extra_body=m["extra_body"])
                    r = common.chat_stream(engines.base_url(cfg) + "/v1",
                                           engines.request_model(cfg, m),
                                           [{"role": "user", "content": "hi"}], req)
                    self.assertEqual(r["error"], "")
                    self.assertEqual(r["output_tokens"], 8)
                    self.assertIsNotNone(r["ttft_s"])
                    self.assertEqual(r["reasoning_chars"], 0)
                self.assertIsNotNone(srv.proc.poll(), "server must be stopped")
        finally:
            if old is None:
                os.environ.pop("BENCH_FAKE_SERVER", None)
            else:
                os.environ["BENCH_FAKE_SERVER"] = old


def fake_study(tmp, family="qwen", machine="dell", engine=None, port=None):
    """A study config for one small model, with a fake HF cache holding its 'weights'."""
    cfg = plan_cfg(family, machine)
    if engine:
        cfg["backend"]["type"] = engine
    cfg["backend"]["port"] = port or free_port()
    cfg["models"] = cfg["models"][:1]
    cfg["sweep"]["idle_baseline_s"] = 0
    m = cfg["models"][0]
    data = b"fake weights"
    sha = hashlib.sha256(data).hexdigest()
    base = pathlib.Path(tmp) / "hub" / ("models--" + m["repo"].replace("/", "--"))
    (base / "blobs").mkdir(parents=True)
    (base / "blobs" / sha).write_bytes(data)
    snap = base / "snapshots" / m["revision"]
    snap.mkdir(parents=True)
    os.symlink(base / "blobs" / sha, snap / "model.safetensors")
    m["weight_files"] = {"model.safetensors": {"sha256": sha, "size": len(data)}}
    path = pathlib.Path(tmp) / "study.json"
    common.write_json(path, cfg)
    return path, cfg


class EndToEndTest(unittest.TestCase):
    """run_bench.py against the fake server: one model, one level, both engines."""

    def run_bench(self, tmp, cfg_path, *extra):
        env = dict(os.environ, BENCH_FAKE_SERVER=FAKE, BENCH_RESULTS=str(pathlib.Path(tmp) / "r"),
                   HF_HUB_CACHE=str(pathlib.Path(tmp) / "hub"))
        cmd = [sys.executable, str(BENCH / "harness/run_bench.py"), "--config", str(cfg_path),
               "--levels", "2", "--num-parallel", "2", "--reps", "1", "--max-tokens", "8",
               "--skip-pull", "--skip-quality", "--allow-deviation", "--ignore-preflight",
               "--no-report", *extra]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
        runs = [d for d in (pathlib.Path(tmp) / "r").glob("*/*") if (d / "env.json").exists()]
        self.assertEqual(len(runs), 1)
        return runs[0], p.stdout

    def check_rows(self, run, cfg, model_name):
        with open(run / "requests.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2 * 1 * len(protocol.WORKLOADS))
        r = rows[0]
        m = cfg["models"][0]
        self.assertEqual(r["protocol_version"], protocol.VERSION)
        self.assertEqual(r["engine"], cfg["backend"]["type"])
        self.assertEqual(r["slots"], "2")
        self.assertEqual(r["model_repo"], m["repo"])
        self.assertEqual(r["model_revision"], m["revision"])
        self.assertEqual(r["quantization"], cfg["quantization"])
        self.assertTrue(all(x["error"] == "" and x["output_tokens"] == "8" for x in rows))
        env = common.load_json(run / "env.json")
        self.assertEqual(env["slots"], 2)
        self.assertEqual(common.slots_of(env), "2")
        self.assertTrue(env["models"][0]["ok"])
        run_j = common.load_json(run / "run.json")
        self.assertEqual(run_j["servers"][0]["command"][-0:][:1][0], sys.executable)
        self.assertFalse((BENCH / ".run" / "bench_server.pid").exists())
        return rows, run_j

    def test_vllm_study(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, cfg = fake_study(tmp)
            run, _ = self.run_bench(tmp, path)
            rows, run_j = self.check_rows(run, cfg, cfg["models"][0]["repo"])
            self.assertEqual(rows[0]["kv_cache_tokens"], "16000")
            self.assertEqual(rows[0]["engine_version"], "0.0-fake")

    def test_mlx_study_with_a_model_that_cannot_turn_thinking_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, cfg = fake_study(tmp, family="gpt-oss", machine="mac:128")
            self.assertEqual(cfg["backend"]["type"], "mlx")
            run, _ = self.run_bench(tmp, path)
            rows, run_j = self.check_rows(run, cfg, "default_model")
            self.assertGreater(sum(int(r["reasoning_chars"]) for r in rows), 0)
            self.assertEqual(rows[0]["kv_cache_dtype"], "n/a (mlx)")

    def test_validator_flags_the_test_run_as_deviating(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, cfg = fake_study(tmp)
            run, _ = self.run_bench(tmp, path)
            env = dict(os.environ, BENCH_RESULTS=str(pathlib.Path(tmp) / "r"))
            p = subprocess.run([sys.executable, str(BENCH / "harness/validate_results.py"),
                                run.parent.name], env=env, capture_output=True, text=True)
            self.assertEqual(p.returncode, 1)
            self.assertIn("deviates", p.stdout)
            self.assertNotIn("missing columns", p.stdout)

    def test_report_builds(self):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            self.skipTest("openpyxl not installed in this interpreter")
        with tempfile.TemporaryDirectory() as tmp:
            path, cfg = fake_study(tmp)
            run, _ = self.run_bench(tmp, path)
            p = subprocess.run([sys.executable, str(BENCH / "harness/make_report.py"), str(run)],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            self.assertTrue((run / "findings.xlsx").exists())


ORCHESTRATE = """
import runpy, sys
sys.path.insert(0, sys.argv[1])
import preflight
preflight.check = lambda *a, **k: {"ok": True, "problems": [], "info": {}}
sys.argv = [sys.argv[2]] + sys.argv[3:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


class OrchestratorTest(unittest.TestCase):
    def test_calibration_resume_and_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, cfg = fake_study(tmp)
            env = dict(os.environ, BENCH_FAKE_SERVER=FAKE,
                       BENCH_RESULTS=str(pathlib.Path(tmp) / "r"),
                       HF_HUB_CACHE=str(pathlib.Path(tmp) / "hub"))
            cmd = [sys.executable, "-c", ORCHESTRATE, str(BENCH / "harness"),
                   str(BENCH / "harness/run_protocol.py"), "--config", str(path),
                   "--calibrate", "--levels", "1", "2", "--free-min", "0", "--test-overrides",
                   "--max-tokens", "8", "--skip-pull", "--allow-deviation", "--ignore-preflight"]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
            self.assertIn("PROTOCOL_DONE", p.stdout)
            study = pathlib.Path(tmp) / "r" / cfg["study"]["id"]
            self.assertTrue((study / "combined" / "findings.xlsx").exists(), p.stdout[-2000:])
            # Resume: both levels are complete, so nothing new runs.
            n_runs = len([d for d in study.iterdir() if (d / "env.json").exists()])
            p2 = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(p2.returncode, 0, p2.stdout[-3000:])
            self.assertNotIn("START", p2.stdout)
            self.assertEqual(n_runs, len([d for d in study.iterdir()
                                          if (d / "env.json").exists()]))


if __name__ == "__main__":
    unittest.main()
