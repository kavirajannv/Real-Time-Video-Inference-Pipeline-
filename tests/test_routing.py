import logging

from vpipe.aggregator import Aggregator
from vpipe.routing import should_route, verify_route_labels
from vpipe.schema import BStatus, SKIP_REASON, Stage, WorkerResult

from conftest import run_fake


def a_result(fid, label, conf, ok=True):
    return WorkerResult(Stage.A, fid, "A0", ok=ok, output={"label": label, "conf": conf}, t_start=1.0, t_end=1.01)


def test_threshold_boundary_inclusive():
    assert should_route("person", 0.30, ["person"], 0.30)[0] is True
    assert should_route("person", 0.2999, ["person"], 0.30)[0] is False
    assert should_route("person", 0.99, ["person"], 0.30)[0] is True


def test_non_target_label_never_routes():
    routed, reason = should_route("car", 0.99, ["person"], 0.30)
    assert routed is False and reason == SKIP_REASON


def test_missing_result_never_routes():
    assert should_route(None, None, ["person"], 0.3)[0] is False


def test_configurable_label_set():
    labels = ["person", "groom", "scuba_diver"]
    assert should_route("groom", 0.5, labels, 0.3)[0] is True
    assert should_route("scuba_diver", 0.29, labels, 0.3)[0] is False


def test_label_verification_against_model_names(caplog):
    names = {0: "tench", 1: "groom", 2: "scuba_diver"}
    with caplog.at_level(logging.WARNING):
        present, missing = verify_route_labels(["person", "groom"], names)
    assert present == ["groom"] and missing == ["person"]
    assert "person" in caplog.text


def test_only_routed_frames_enter_model_b_queue():
    agg = Aggregator(["person"], 0.30, 10, 10)
    cases = {0: ("person", 0.76), 1: ("car", 0.95), 2: ("person", 0.29), 3: ("person", 0.30)}
    for fid, (label, conf) in cases.items():
        agg.on_ingest(fid, 0.0, 0.0)
        agg.on_result(a_result(fid, label, conf))
    assert list(agg.pending_b) == [0, 3]
    assert agg.records[1].b_status == BStatus.SKIPPED and agg.records[1].route_reason == SKIP_REASON
    assert agg.records[2].b_status == BStatus.SKIPPED


def test_failed_classification_is_not_routed():
    agg = Aggregator(["person"], 0.30, 10, 10)
    agg.on_ingest(0, 0.0, 0.0)
    agg.on_result(WorkerResult(Stage.A, 0, "A0", ok=False, error="boom"))
    assert not agg.pending_b
    assert agg.records[0].b_status == BStatus.NOT_RUN


def test_pipeline_routes_exactly_the_matching_frames(tmp_path):
    # fake classifier: id%3==0 -> person 0.90 (route), id%3==1 -> car, id%3==2 -> person 0.20 (below threshold)
    summary, rows = run_fake(tmp_path, n=30)
    for r in rows:
        assert r["routed"] == (r["frame_id"] % 3 == 0), r
        if not r["routed"]:
            assert r["b_status"] == "SKIPPED" and r["b_worker"] is None and r["b_infer_ms"] is None
    assert summary["frames_routed_to_b"] == 10
    assert summary["frames_skipped"] == 20


def test_detection_stride_sends_every_nth_routed_frame_to_model_b():
    from vpipe.schema import STRIDE_REASON

    agg = Aggregator(["person"], 0.30, 10, 10, detect_every=2)
    for fid in range(6):
        agg.on_ingest(fid, 0.0, 0.0)
        agg.on_result(a_result(fid, "person" if fid != 4 else "car", 0.9))
    assert list(agg.pending_b) == [0, 2]  # 4 is not a person frame at all
    assert agg.records[1].route_reason.startswith(STRIDE_REASON)
    assert agg.records[4].route_reason == SKIP_REASON  # a real "no" from Model A stays a real "no"


def test_not_gated_frame_goes_straight_to_model_b():
    from vpipe.schema import AStatus

    agg = Aggregator(["person"], 0.30, 10, 10)
    agg.on_ingest(0, 0.0, None, AStatus.NOT_GATED, "fast mode: no gate")
    assert list(agg.pending_b) == [0] and agg.records[0].routed
    assert not agg.records[0].is_failure


def test_fast_mode_pipeline_holds_and_skips_the_gate(tmp_path):
    # every 3rd frame goes straight to Model B; the others are HELD; nothing is a failure or missing
    summary, rows = run_fake(tmp_path, n=30, infer_every=3, gate=False)
    assert [r["frame_id"] for r in rows] == list(range(30))
    for r in rows:
        if r["frame_id"] % 3 == 0:
            assert r["a_status"] == "NOT_GATED" and r["routed"] and r["b_status"] in ("DETECTIONS", "NO_DETECTIONS")
        else:
            assert r["a_status"] == "HELD" and r["b_status"] == "SKIPPED" and r["status"] == "OK"
    assert summary["frames_held"] == 20 and summary["frames_not_gated"] == 10


def test_fast_settings_scale_with_cpu_size():
    from vpipe.cli import fast_settings

    assert fast_settings(4) == (8, 2)
    assert fast_settings(6) == (6, 3)
    assert fast_settings(8)[0] < fast_settings(4)[0]
