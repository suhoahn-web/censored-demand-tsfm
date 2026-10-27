"""배양육(JMM r1 32B) 재개 실행을 기록 규약대로 안전하게 중단한다 (lab06에서 실행).

절차 (interruptions/README_resume_20261022.md의 10-22 중단 방식과 동일):
  1. 중단 전 백업(read-only): responses_resumed.jsonl, status.json, manifest
  2. status.json이 갱신되는 순간(배치 플러시 직후)에 SIGTERM
  3. 종료 확인 후 검증: 줄 수 ≥ status.responses_written, 마지막 줄 완전한 JSON,
     bad JSON 0, sha256 기록
  4. 중단 후 백업(read-only) + interruptions/*_interruption_20261027.json + README

재개: README의 production 명령에 --continue-resume 추가 (큐 스크립트 말미에서 자동 실행).
사용: /home/ubuntu/miniconda3/envs/csm/bin/python jmm_safe_stop_20261027.py --pid 54910
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/data/cultured_meat_jmm_r1")
RUN = BASE / "runs/full641_32b_direct_repetition30_20261011_v1"
OUT = BASE / "runs/full641_32b_direct_repetition30_20261011_v1_resume_20261022"
RESUMED = OUT / "responses_resumed.jsonl"
STATUS = OUT / "status.json"
TAG = "20261027"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def backup(label: str) -> Path:
    d = BASE / "backups" / f"{OUT.name}_{label}_{TAG}"
    d.mkdir(parents=True, exist_ok=False)
    for f in [RESUMED, STATUS] + sorted(OUT.glob("resume_manifest_*.json")) + [OUT / "technical_check.json"]:
        if f.exists():
            shutil.copy2(f, d / f.name)
    for f in d.iterdir():
        os.chmod(f, 0o440)
    os.chmod(d, 0o550)
    return d


def verify() -> dict:
    n = bad = 0
    last = None
    with open(RESUMED, "rb") as f:
        for raw in f:
            n += 1
            last = raw
            try:
                json.loads(raw)
            except Exception:
                bad += 1
    last_complete = bool(last) and last.endswith(b"\n") and bad == 0
    return {"responses_lines": n, "bad_json_lines": bad, "last_line_complete": last_complete,
            "responses_sha256": sha(RESUMED)}


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--max-wait-flush", type=int, default=900, help="status 갱신 대기 상한(초)")
    a = ap.parse_args()
    cmdline = Path(f"/proc/{a.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert "jmm_r1_resume_32b.py" in cmdline, cmdline
    st0 = json.loads(STATUS.read_text())
    assert st0.get("pid") == a.pid and st0.get("state") == "generating", st0

    pre = backup("prestop")
    print("prestop backup:", pre, flush=True)

    # 플러시 직후를 노린다: status.json의 updated_at이 바뀌는 순간 SIGTERM
    t0 = time.time()
    last_upd = st0["updated_at_utc"]
    while time.time() - t0 < a.max_wait_flush:
        st = json.loads(STATUS.read_text())
        if st["updated_at_utc"] != last_upd:
            break
        time.sleep(0.5)
    else:
        print("WARNING: status.json not updated within wait window; sending SIGTERM anyway", flush=True)
    os.kill(a.pid, signal.SIGTERM)
    t_sig = now()
    print("SIGTERM sent at", t_sig, "status at signal:", json.dumps(st), flush=True)
    for i in range(600):
        if not alive(a.pid):
            break
        time.sleep(1)
        if i == 300:
            os.kill(a.pid, signal.SIGTERM)
            print("second SIGTERM at 300 s", flush=True)
    if alive(a.pid):
        os.kill(a.pid, signal.SIGKILL)
        print("SIGKILL after 600 s", flush=True)
        method = "SIGTERM (x2) then SIGKILL after 600 s"
    else:
        method = "SIGTERM sent to PID %d immediately after status.json update (post-flush)" % a.pid
    time.sleep(5)
    procs = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                    text=True).strip()
    print("GPU compute procs after stop:", repr(procs), flush=True)

    v = verify()
    st_final = json.loads(STATUS.read_text())
    post = backup("poststop")
    rec = {
        "interrupted_at_utc": t_sig,
        "run_dir": str(RUN), "resume_dir": str(OUT),
        "reason": "GPU reallocation to JORS revision experiments (TSFM censored-demand paper, R1); "
                  "not a technical failure of this run",
        "method": method,
        "status_json_at_interruption": st_final,
        **v,
        "lines_vs_status": v["responses_lines"] - int(st_final.get("responses_written", 0)),
        "note": "lines may exceed status.responses_written by at most one batch written after the last status "
                "update; replay() excludes an incomplete trailing batch and regenerates it",
        "backups": [str(pre), str(post)],
        "resume_command": (
            "cd /data/cultured_meat_jmm_r1 && nohup /home/ubuntu/miniconda3/envs/csm/bin/python "
            "jmm_r1_resume_32b.py production --run runs/full641_32b_direct_repetition30_20261011_v1 "
            "--out runs/full641_32b_direct_repetition30_20261011_v1_resume_20261022 "
            "--interruption-record interruptions/full641_32b_direct_repetition30_20261011_v1_interruption_20261022.json "
            "--continue-resume >> runs/full641_32b_direct_repetition30_20261011_v1_resume_20261022_continue_20261027.log 2>&1 &"),
        "resume_policy": "same resume folder, --continue-resume (replay verifies original + resumed records, "
                         "excludes any partial trailing batch, continues from next batch); original run folder untouched",
    }
    rp = BASE / "interruptions" / f"{RUN.name}_interruption2_{TAG}.json"
    rp.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    readme = BASE / "interruptions" / f"README_resume2_{TAG}.md"
    readme.write_text(f"""# JMM r1 32B resume run: second interruption {TAG} and how to continue

Resume run folder: {OUT}  (first resume, started 2026-10-22)
Stopped {t_sig} by {method}. Reason: GPU reallocated to the JORS R1 experiments of the TSFM censored-demand paper.
Records at stop: {v['responses_lines']} lines in responses_resumed.jsonl (status.responses_written = {st_final.get('responses_written')}),
bad JSON {v['bad_json_lines']}, last line complete: {v['last_line_complete']}, sha256 {v['responses_sha256'][:12]}...
Backups (read-only): {pre.name}, {post.name}
Interruption record: {rp.name}

## To continue (GPU must be free; the script refuses otherwise):
{rec['resume_command']}
# Progress: cat {OUT}/status.json
""")
    print(json.dumps(rec, indent=2)[:1500], flush=True)
    print("record:", rp, flush=True)


if __name__ == "__main__":
    main()
