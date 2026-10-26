from vpipe.models import fake_classification

from conftest import run_fake, video_info


def _contiguous(rows, n):
    assert [r["frame_id"] for r in rows] == list(range(n))


def test_model_a_exception_is_reported_not_faked(tmp_path, fault):
    n = 30
    summary, rows = run_fake(tmp_path, n=n, fault_a=fault(error_every=5))
    _contiguous(rows, n)
    for r in rows:
        if r["frame_id"] % 5 == 0:
            assert r["a_status"] == "ERROR" and "injected" in r["a_error"]
            assert r["a_label"] is None and r["routed"] is False and r["b_status"] == "NOT_RUN"
            assert r["status"] == "ERROR"
        else:
            assert r["a_status"] == "OK" and r["a_label"] == fake_classification(r["frame_id"])[0]
    assert summary["a_errors"] == 6


def test_model_b_exception_is_reported(tmp_path, fault):
    n = 30
    summary, rows = run_fake(tmp_path, n=n, fault_b=fault(error_every=2))  # every even routed frame fails
    _contiguous(rows, n)
    for r in rows:
        if r["routed"] and r["frame_id"] % 2 == 0:
            assert r["b_status"] == "ERROR" and r["detections"] == [] and r["status"] == "ERROR"
    assert summary["b_errors"] == 5  # routed: 0,3,...,27 -> even ones 0,6,12,18,24


def test_slow_model_a_times_out_and_late_result_is_discarded(tmp_path, fault):
    n = 20
    # 3 workers: two can be stuck on slow frames while the third keeps the rest flowing
    summary, rows = run_fake(tmp_path, n=n, a_workers=3, timeout_a_s=0.4,
                             fault_a=fault(delay_s=1.0, delay_every=10))
    _contiguous(rows, n)
    slow = [r for r in rows if r["frame_id"] % 10 == 0]
    assert all(r["a_status"] == "TIMEOUT" and r["b_status"] == "NOT_RUN" for r in slow)
    assert all(r["a_status"] == "OK" for r in rows if r["frame_id"] % 10 != 0)
    assert summary["a_timeouts"] == 2


def test_slow_model_b_times_out_without_blocking_model_a(tmp_path, fault):
    n = 24
    summary, rows = run_fake(tmp_path, n=n, b_workers=3, timeout_b_s=0.4, fault_b=fault(delay_s=1.0, delay_every=12))
    _contiguous(rows, n)
    for r in rows:
        assert r["a_status"] == "OK"  # Model A unaffected
        if r["frame_id"] % 12 == 0:
            assert r["b_status"] == "TIMEOUT"
        elif r["routed"]:
            assert r["b_status"] in ("DETECTIONS", "NO_DETECTIONS")
    assert summary["b_timeouts"] == 2


def test_worker_crash_is_detected_and_worker_respawned(tmp_path, fault):
    n = 40
    summary, rows = run_fake(tmp_path, n=n, b_workers=1, timeout_b_s=20.0, fault_b=fault(crash_after=2))
    _contiguous(rows, n)
    crashed = [r for r in rows if r["b_status"] == "CRASHED"]
    assert len(crashed) == 1 and "died" in crashed[0]["b_error"]
    assert summary["worker_restarts"]["B"] == 1
    # after the respawn the remaining routed frames are processed normally
    later = [r for r in rows if r["routed"] and r["frame_id"] > crashed[0]["frame_id"]]
    assert later and all(r["b_status"] in ("DETECTIONS", "NO_DETECTIONS") for r in later)


def test_invalid_frames_are_kept_and_marked(tmp_path):
    n = 20
    summary, rows = run_fake(tmp_path, n=n, invalid_every=7)
    _contiguous(rows, n)
    invalid = [r["frame_id"] for r in rows if r["a_status"] == "INVALID"]
    assert invalid == [6, 13]
    assert summary["frames_invalid"] == 2
    assert video_info(tmp_path / "output_annotated.mp4")[0] == n


def test_second_run_into_a_busy_output_folder_is_refused(tmp_path):
    import os

    import pytest

    from vpipe.pipeline import PipelineError

    (tmp_path / ".vpipe.lock").write_text(str(os.getppid()))  # a live process "owns" the folder
    with pytest.raises(PipelineError, match="another vpipe run"):
        run_fake(tmp_path, n=5)
    (tmp_path / ".vpipe.lock").write_text("999999999")  # stale lock from a killed run is taken over
    summary, rows = run_fake(tmp_path, n=5)
    assert len(rows) == 5 and not (tmp_path / ".vpipe.lock").exists()
