"""Run the v3 batch driver in one fresh Blender process per scene.

The v3 batch driver itself owns scene acceptance and output integrity.  This
outer supervisor only bounds Blender lifetime to one scene at a time so the
8-GB workstation does not accumulate Cycles/Blender allocations across a
500-scene run.  It never starts two Blender processes concurrently.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLENDER_DEFAULT = "/Applications/Blender.app/Contents/MacOS/Blender"
TEMPLATE_DEFAULT = ROOT / "reports" / "v3_scan_cache.blend"
DRIVER = ROOT / "tools" / "run_500_scenes.py"
RECOVERY_DRIVER = ROOT / "tools" / "recover_scene_v3_batch.py"


def accepted_count(output_root: Path) -> int:
    return sum(1 for path in (output_root / "states").glob("scene_*/accepted.json") if path.is_file())


def scene_is_accepted(output_root: Path, scene_number: int) -> bool:
    checkpoint = output_root / "states" / f"scene_{scene_number:04d}" / "accepted.json"
    if not checkpoint.is_file():
        return False
    try:
        return json.loads(checkpoint.read_text(encoding="utf-8")).get("status") == "accepted"
    except (OSError, json.JSONDecodeError):
        return False


def run_blender_command(command: list[str], log_path: Path, output_root: Path, scene_number: int, stop_after_accepted: bool = False) -> subprocess.CompletedProcess:
    """Run one child, with a bounded post-checkpoint exit grace period."""

    started = time.monotonic()
    accepted_since = None
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while True:
            returncode = process.poll()
            if returncode is not None:
                return subprocess.CompletedProcess(command, returncode)
            if stop_after_accepted and scene_is_accepted(output_root, scene_number):
                if accepted_since is None:
                    accepted_since = time.monotonic()
                elif time.monotonic() - accepted_since >= 15:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    return subprocess.CompletedProcess(command, 0)
            if time.monotonic() - started >= 1800:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                return subprocess.CompletedProcess(command, 1)
            time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v3_500_scenes.json"))
    parser.add_argument("--template", default=str(TEMPLATE_DEFAULT))
    parser.add_argument("--blender", default=BLENDER_DEFAULT)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--log", default=str(ROOT / "reports" / "serial_blender_batch.jsonl"))
    args = parser.parse_args()
    if not 1 <= args.start <= args.end <= 500:
        raise SystemExit("scene range must be within 1..500")
    config = json.loads(Path(args.config).read_text())
    output_root = ROOT / config["output_root"]
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    started = time.time()
    for scene_number in range(args.start, args.end + 1):
        command = [
            args.blender,
            "--background",
            str(Path(args.template).resolve()),
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python",
            str(DRIVER),
            "--",
            "--config",
            str(Path(args.config).resolve()),
            "--start",
            str(scene_number),
            "--end",
            str(scene_number),
            "--template-cache",
        ]
        scene_started = time.time()
        result = run_blender_command(command, log_path, output_root, scene_number)
        accepted = scene_is_accepted(output_root, scene_number)
        recovery_result = None
        recovery_command = None
        if result.returncode != 0 or not accepted:
            recovery_command = list(command)
            recovery_command[recovery_command.index(str(DRIVER))] = str(RECOVERY_DRIVER)
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\nV3_BATCH_RECOVERY_START " + json.dumps({"scene_number": scene_number, "reason": "initial_process_failed_or_no_accepted_checkpoint"}, ensure_ascii=False) + "\n")
            recovery_result = run_blender_command(recovery_command, log_path, output_root, scene_number, stop_after_accepted=True)
            accepted = scene_is_accepted(output_root, scene_number)
        row = {
            "scene_number": scene_number,
            "returncode": result.returncode,
            "final_returncode": recovery_result.returncode if recovery_result is not None else result.returncode,
            "accepted": accepted,
            "recovery_attempted": recovery_result is not None,
            "recovery_returncode": recovery_result.returncode if recovery_result is not None else None,
            "seconds": time.time() - scene_started,
            "accepted_checkpoint_count": accepted_count(output_root),
            "command": command,
        }
        if recovery_command is not None:
            row["recovery_command"] = recovery_command
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print("SERIAL_BLENDER_SCENE " + json.dumps(row, ensure_ascii=False), flush=True)
        if not accepted or (recovery_result is not None and recovery_result.returncode != 0):
            summary = {"status": "failed", "start": args.start, "end": args.end, "failed_scene": scene_number, "returncode": result.returncode, "accepted": accepted, "seconds": time.time() - started, "accepted_checkpoint_count": accepted_count(output_root), "log": str(log_path)}
            (ROOT / "reports" / "serial_blender_batch.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return result.returncode or 1
    summary = {"status": "complete", "start": args.start, "end": args.end, "seconds": time.time() - started, "accepted_checkpoint_count": accepted_count(output_root), "log": str(log_path)}
    (ROOT / "reports" / "serial_blender_batch.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SERIAL_BLENDER_COMPLETE " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
