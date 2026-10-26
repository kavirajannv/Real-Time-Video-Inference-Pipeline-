import json
import time

from vpipe.config import ModelSpec

from conftest import run_fake, video_info


def test_offline_backpressure_never_drops_and_stays_bounded(tmp_path):
    n = 40
    summary, rows = run_fake(
        tmp_path, n=n, a_workers=1, b_workers=1, queue_a_size=2, queue_b_size=1, max_buffered_frames=6,
        model_a=ModelSpec(kind="fake", fake_infer_s=0.01), model_b=ModelSpec(kind="fake", fake_infer_s=0.03),
    )
    assert [r["frame_id"] for r in rows] == list(range(n))
    assert summary["frames_dropped"] == 0
    rt = summary["runtime"]
    assert rt["queue_a_depth"]["max"] <= 2
    assert rt["queue_b_depth"]["max"] <= 1
    assert rt["frames_buffered"]["max"] <= 6
    assert summary["reader_blocked_s"] > 0  # the reader really was throttled


def test_live_mode_drops_and_records_frames_when_overloaded(tmp_path):
    n = 60
    summary, rows = run_fake(
        tmp_path, n=n, mode="live", live_fps=200.0, a_workers=1, b_workers=1, queue_a_size=2,
        model_a=ModelSpec(kind="fake", fake_infer_s=0.03), model_b=ModelSpec(kind="fake", fake_infer_s=0.01),
    )
    assert [r["frame_id"] for r in rows] == list(range(n))  # dropped frames still get an output frame
    dropped = [r for r in rows if r["a_status"] == "DROPPED"]
    assert dropped and summary["frames_dropped"] == len(dropped)
    assert all(r["status"] == "DROPPED" and r["b_status"] == "NOT_RUN" for r in dropped)
    assert 0 < summary["drop_rate"] < 1
    assert video_info(tmp_path / "output_annotated.mp4")[0] == n


def test_end_of_video_with_frames_in_flight_flushes_everything(tmp_path):
    # last frames are still inside slow Model B workers when the reader hits EOF
    n = 12
    summary, rows = run_fake(tmp_path, n=n, model_b=ModelSpec(kind="fake", fake_infer_s=0.3))
    assert [r["frame_id"] for r in rows] == list(range(n))
    assert all(r["status"] == "OK" for r in rows)
    assert video_info(tmp_path / "output_annotated.mp4")[0] == n


def test_stop_during_processing_shuts_down_cleanly(tmp_path):
    t = time.perf_counter()
    summary, rows = run_fake(
        tmp_path, n=500, stop_after_s=1.0,
        model_a=ModelSpec(kind="fake", fake_infer_s=0.02), model_b=ModelSpec(kind="fake", fake_infer_s=0.2),
    )
    elapsed = time.perf_counter() - t
    assert elapsed < 30, "shutdown took too long (deadlock?)"
    assert summary["aborted"] is True
    k = len(rows)
    assert 0 < k < 500
    assert [r["frame_id"] for r in rows] == list(range(k))  # everything ingested is accounted for, in order
    assert summary["frames_incomplete"] > 0
    assert all(r["status"] in ("OK", "INCOMPLETE") for r in rows)
    # files were finalised
    assert len(json.loads((tmp_path / "frames_log.json").read_text())) == k
    assert video_info(tmp_path / "output_annotated.mp4")[0] == k
