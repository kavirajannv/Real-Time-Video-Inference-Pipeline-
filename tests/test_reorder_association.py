import random

from vpipe.aggregator import Aggregator
from vpipe.models import fake_classification, fake_detections
from vpipe.schema import AStatus, BStatus, Stage, WorkerResult

from conftest import run_fake, video_info


def _a(fid, label="person", conf=0.9):
    return WorkerResult(Stage.A, fid, "A0", ok=True, output={"label": label, "conf": conf}, t_start=0, t_end=0)


def _b(fid, dets):
    return WorkerResult(Stage.B, fid, "B0", ok=True, output=dets, t_start=0, t_end=0)


def test_out_of_order_completion_is_released_in_order():
    agg = Aggregator(["person"], 0.3, 10, 10)
    n = 50
    for fid in range(n):
        agg.on_ingest(fid, 0.0, 0.0)
    agg.set_total(n)
    order = list(range(n))
    random.Random(1).shuffle(order)
    released = []
    for fid in order:
        agg.on_result(_a(fid, "car", 0.9))  # not routed -> final immediately
        released += [r.frame_id for r in agg.pop_ready()]
    assert released == list(range(n))
    assert agg.done()


def test_frame_waits_for_model_b_before_release():
    agg = Aggregator(["person"], 0.3, 10, 10)
    for fid in range(3):
        agg.on_ingest(fid, 0.0, 0.0)
    agg.on_result(_a(0, "person", 0.9))  # routed, B pending
    agg.on_result(_a(1, "car", 0.9))
    agg.on_result(_a(2, "car", 0.9))
    assert agg.pop_ready() == []           # 1 and 2 are done but 0 is not
    assert agg.reorder_depth == 2
    agg.mark_b_dispatched(0, 0.0, 2.0)
    agg.on_result(_b(0, [{"label": "person", "conf": 0.8, "box": [1, 2, 3, 4]}]))
    out = agg.pop_ready()
    assert [r.frame_id for r in out] == [0, 1, 2]
    assert out[0].detections[0]["box"] == [2, 4, 6, 8]  # scaled back to original resolution


def test_result_before_ingest_event_is_parked_not_lost():
    agg = Aggregator(["person"], 0.3, 10, 10)
    agg.on_result(_a(0, "car", 0.5))
    assert 0 not in agg.records
    agg.on_ingest(0, 0.0, 0.0)
    assert agg.records[0].a_status == AStatus.OK and agg.records[0].a_label == "car"


def test_late_result_cannot_overwrite_timeout():
    clock = [0.0]
    agg = Aggregator(["person"], 0.3, 1.0, 1.0, clock=lambda: clock[0])
    agg.on_ingest(0, 0.0, 0.0)
    clock[0] = 5.0
    assert agg.check_timeouts() == 1
    agg.on_result(_a(0, "person", 0.99))
    rec = agg.records[0]
    assert rec.a_status == AStatus.TIMEOUT and rec.a_label is None and not rec.routed
    assert agg.late_results == 1


def test_model_b_result_for_unrouted_frame_is_rejected():
    agg = Aggregator(["person"], 0.3, 10, 10)
    agg.on_ingest(0, 0.0, 0.0)
    agg.on_result(_a(0, "car", 0.9))
    agg.on_result(_b(0, [{"label": "x", "conf": 1.0, "box": [0, 0, 1, 1]}]))
    assert agg.records[0].detections == []
    assert agg.unexpected_results == 1


def test_routed_with_no_detections_differs_from_skipped():
    agg = Aggregator(["person"], 0.3, 10, 10)
    agg.on_ingest(0, 0.0, 0.0)
    agg.on_ingest(1, 0.0, 0.0)
    agg.on_result(_a(0, "person", 0.9))
    agg.mark_b_dispatched(0, 0.0, 1.0)
    agg.on_result(_b(0, []))
    agg.on_result(_a(1, "car", 0.9))
    assert agg.records[0].b_status == BStatus.NO_DETECTIONS
    assert agg.records[1].b_status == BStatus.SKIPPED


def test_pipeline_results_match_their_frames_under_jitter(tmp_path, fault):
    """Random per-task delays on 2+2 workers force out-of-order completion.

    Every synthetic frame encodes its own id in its pixels and the fake models derive their
    output from those pixels, so any mix-up between frames shows up as a label mismatch.
    """
    n = 60
    summary, rows = run_fake(tmp_path, n=n, fault_a=fault(jitter_s=0.02, seed=1), fault_b=fault(jitter_s=0.03, seed=2))
    assert [r["frame_id"] for r in rows] == list(range(n))
    for r in rows:
        fid = r["frame_id"]
        label, conf = fake_classification(fid)
        assert (r["a_label"], r["a_conf"]) == (label, conf)
        if r["routed"]:
            expected = fake_detections(fid)
            assert [d["label"] for d in r["detections"]] == [d["label"] for d in expected]
            assert r["b_status"] == ("DETECTIONS" if expected else "NO_DETECTIONS")
    # both workers of each pool actually did work
    assert {r["a_worker"] for r in rows} == {"A0", "A1"}
    assert {r["b_worker"] for r in rows if r["routed"]} == {"B0", "B1"}

    frames, shape = video_info(tmp_path / "output_annotated.mp4")
    assert frames == n and shape == (120, 160, 3)
