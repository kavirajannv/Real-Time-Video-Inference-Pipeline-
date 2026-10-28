import json

import cv2
import numpy as np

from vpipe.analytics import Analytics, SceneCutDetector
from vpipe.schema import AStatus, BStatus, FrameRecord

W, H = 640, 360


def _frame(colour=(90, 90, 90)):
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:] = colour
    img[: H // 2, : W // 2] = (200, 50, 50)  # some structure for the histogram
    return img


def _rec(fid, dets, b_status=BStatus.DETECTIONS):
    r = FrameRecord(frame_id=fid, t_ingest=0.0, a_status=AStatus.OK, b_status=b_status, routed=True)
    r.detections = dets
    return r


def _person(cx, foot_y, h=120, conf=0.9):
    return {"label": "person", "cls": 0, "conf": conf, "box": [cx - 25, foot_y - h, cx + 25, foot_y]}


def test_person_walking_down_crosses_line_once_and_is_counted_once(tmp_path):
    an = Analytics(tmp_path, (W, H), 30.0, count_line=(0, 0.75, 1, 0.75), min_track_frames=5)
    img = _frame()
    line_y = 0.75 * H
    ys = np.linspace(line_y - 60, line_y + 60, 30)
    for i, y in enumerate(ys):
        out = an.process(_rec(i, [_person(300, float(y))]), img)
    assert out["people"] == 1
    assert out["line_in"] == 1 and out["line_out"] == 0
    assert out["unique_people"] == 1
    assert len({t["id"] for t in out["tracks"]}) == 1


def test_two_people_opposite_directions(tmp_path):
    an = Analytics(tmp_path, (W, H), 30.0, min_track_frames=5)
    img = _frame()
    line_y = 0.75 * H
    for i in range(30):
        down = line_y - 60 + 4 * i
        up = line_y + 60 - 4 * i
        out = an.process(_rec(i, [_person(150, down), _person(500, up)]), img)
    assert out["people"] == 2 and out["unique_people"] == 2
    assert out["line_in"] == 1 and out["line_out"] == 1


def test_short_lived_tracks_are_not_unique_people(tmp_path):
    an = Analytics(tmp_path, (W, H), 30.0, min_track_frames=8)
    img = _frame()
    for i in range(3):
        an.process(_rec(i, [_person(300, 200)]), img)
    for i in range(3, 10):
        out = an.process(_rec(i, []), img)
    assert out["unique_people"] == 0


def test_failed_frame_reports_unavailable_not_zero(tmp_path):
    an = Analytics(tmp_path, (W, H), 30.0)
    img = _frame()
    an.process(_rec(0, [_person(300, 200)]), img)
    out = an.process(_rec(1, [], b_status=BStatus.TIMEOUT), img)
    assert out["status"] == "UNAVAILABLE" and out["people"] is None


def test_scene_cut_resets_tracking_and_summary_files_are_written(tmp_path):
    an = Analytics(tmp_path, (W, H), 30.0, min_track_frames=5)
    a, b = _frame((90, 90, 90)), _frame((230, 230, 230))
    b[:] = (230, 230, 230)
    for i in range(70):
        an.process(_rec(i, [_person(300, 200)]), a)
    for i in range(70, 140):
        out = an.process(_rec(i, [_person(300, 200)]), b)
    assert out["scene"] == 1
    summary = an.finalize()
    assert len(summary["scenes"]) == 2
    assert summary["people"]["unique_total"] == 2  # same position, different scene -> different person
    for name in ("analytics_summary.json", "counts.csv", "people_over_time.png", "heatmaps.png"):
        assert (tmp_path / name).exists(), name
    assert json.loads((tmp_path / "analytics_summary.json").read_text())["frames"] == 140


def test_scene_cut_detector_ignores_small_changes():
    det = SceneCutDetector(lag=1, min_scene_frames=1)
    img = _frame()
    assert not det.is_cut(img)
    noisy = np.clip(img.astype(int) + np.random.default_rng(0).integers(-8, 8, img.shape), 0, 255).astype(np.uint8)
    assert not det.is_cut(noisy)
    occluded = noisy.copy()
    occluded[:, : W // 3] = 0  # a large foreground object covers a third of the view
    assert not det.is_cut(occluded)
    other = np.zeros_like(img)
    other[:] = (10, 200, 240)
    assert det.is_cut(other)


def test_counting_pauses_during_scene_transition(tmp_path):
    an = Analytics(tmp_path, (W, H), 30.0, min_track_frames=5)
    a, b = _frame((90, 90, 90)), _frame((230, 230, 230))
    b[:] = (230, 230, 230)
    statuses = []
    for i in range(70):
        statuses.append(an.process(_rec(i, [_person(300, 200)]), a)["status"])
    # dissolve: blend A -> B over 25 frames while a "ghost" walks across the counting line
    for k in range(25):
        img = cv2.addWeighted(a, 1 - k / 24, b, k / 24, 0)
        out = an.process(_rec(70 + k, [_person(500, 0.75 * H - 40 + 4 * k)]), img)
        statuses.append(out["status"])
    assert "TRANSITION" in statuses
    assert out["line_in"] == 0 and out["line_out"] == 0  # no crossings invented from blended frames
    assert all(t["id"] is None for t in out["tracks"])   # boxes shown, but no identities assigned


class _ColourAttributes:
    """Fake attribute model: a blue crop is 'male with a bag', a red crop is 'female'."""

    def __init__(self):
        self.calls = 0

    def predict(self, crop):
        self.calls += 1
        blue = crop[..., 0].mean() > crop[..., 2].mean()
        return np.array([0.9 if blue else 0.1, 0.8 if blue else 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)


def _two_colour_frame():
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:, : W // 2] = (220, 60, 40)   # BGR blue-ish left half
    img[:, W // 2:] = (40, 60, 220)    # red-ish right half
    return img


def test_attributes_are_averaged_per_track_and_summarised(tmp_path):
    model = _ColourAttributes()
    an = Analytics(tmp_path, (W, H), 30.0, min_track_frames=5, attributes=model)
    img = _two_colour_frame()
    for i in range(40):
        out = an.process(_rec(i, [_person(150, 250), _person(480, 250)]), img)
    genders = {t["gender"] for t in out["tracks"]}
    assert genders == {"male", "female"}
    assert out["gender_in_frame"] == {"male": 1, "female": 1, "unknown": 0}
    assert model.calls <= 2 * an.attrs.max_samples  # sampled, not run on every frame
    summary = an.finalize()
    pa = summary["people_attributes"]
    assert (pa["male"], pa["female"], pa["unknown"], pa["has_bag"]) == (1, 1, 0, 1)
    for name in ("people_tracks.csv", "people_gallery.png"):
        assert (tmp_path / name).exists(), name


def test_small_or_edge_people_stay_unknown(tmp_path):
    model = _ColourAttributes()
    an = Analytics(tmp_path, (W, H), 30.0, min_track_frames=5, attributes=model)
    img = _two_colour_frame()
    for i in range(40):
        # 40 px tall (too small to classify) and a body cut off by the bottom edge
        out = an.process(_rec(i, [_person(150, 250, h=40), _person(480, H, h=150)]), img)
    assert model.calls == 0
    assert {t["gender"] for t in out["tracks"]} == {"unknown"}
    assert an.finalize()["people_attributes"]["unknown"] == 2


def test_gender_needs_several_samples(tmp_path):
    an = Analytics(tmp_path, (W, H), 30.0, min_track_frames=1, attributes=_ColourAttributes())
    img = _two_colour_frame()
    out = an.process(_rec(0, [_person(150, 250)]), img)
    assert out["tracks"][0]["gender"] == "unknown"  # one crop is not enough evidence


def test_stride_frames_hold_the_last_boxes_without_new_identities(tmp_path):
    from vpipe.schema import STRIDE_REASON

    an = Analytics(tmp_path, (W, H), 30.0, min_track_frames=5, detect_every=2)
    img = _frame()
    for i in range(20):
        if i % 2:
            rec = _rec(i, [], b_status=BStatus.SKIPPED)
            rec.route_reason = f"{STRIDE_REASON}: Model B runs every 2 frames"
        else:
            rec = _rec(i, [_person(300, 200 + i)])
        out = an.process(rec, img)
        assert out["status"] == "OK" and out["people"] == 1, i  # never drops to 0 on a held frame
    assert out["unique_people"] == 1
    assert len({t["id"] for t in out["tracks"]}) == 1


def test_held_boxes_follow_the_tracks_motion(tmp_path):
    from vpipe.schema import STRIDE_REASON

    # a person walking right at 4 px/frame, detected every 5th frame: in between, the box must keep
    # moving with them instead of freezing where the last detection put it
    an = Analytics(tmp_path, (W, H), 30.0, min_track_frames=5, detect_every=5)
    img = _frame()
    errors = []
    for i in range(40):
        cx = 100 + 4 * i
        if i % 5:
            rec = _rec(i, [], b_status=BStatus.SKIPPED)
            rec.route_reason = f"{STRIDE_REASON}: Model B runs every 5 frames"
        else:
            rec = _rec(i, [_person(cx, 250)])
        out = an.process(rec, img)
        if i >= 15 and i % 5:  # velocity known after a few detections
            x1, _, x2, _ = out["tracks"][0]["box"]
            errors.append(abs((x1 + x2) / 2 - cx))
    assert max(errors) < 4.0, errors  # frozen boxes would be 4-16 px behind


def test_people_frames_render_with_frame_confidence_and_separate_labels(tmp_path):
    from vpipe.schema import STRIDE_REASON
    from vpipe.writer import annotate, status_lines

    an = Analytics(tmp_path, (W, H), 30.0, min_track_frames=5, detect_every=2)
    img = _frame()
    crowd = [_person(200 + 30 * j, 250, conf=0.5 + 0.1 * j) for j in range(4)]  # overlapping people
    for i in range(6):
        rec = _rec(i, crowd if i % 2 == 0 else [], b_status=BStatus.DETECTIONS if i % 2 == 0 else BStatus.SKIPPED)
        rec.a_label, rec.a_conf = "person", 0.9
        if i % 2:
            rec.route_reason = f"{STRIDE_REASON}: Model B runs every 2 frames"
        rec.analytics = an.process(rec, img)
        out = annotate(img, rec, an.line)  # every drawing path runs without error
        assert out.shape == img.shape
        box_line = status_lines(rec)[-1][0]
        if i % 2 == 0:
            assert box_line.startswith("Boxes: DETECTED"), box_line
        else:
            assert box_line.startswith("Boxes: estimated (+1 fr)"), box_line
        assert "conf mean" in box_line or i == 0  # a brand-new track may not be confirmed yet


def test_min_conf_shown_hides_weak_boxes_but_keeps_them_counted():
    from vpipe.writer import _box_line, _drawn_tracks

    an = {"status": "OK", "box_source": "detected", "box_age_frames": 0, "people": 3,
          "tracks": [{"id": i, "label": "person", "conf": c, "box": [0, 0, 10, 10]} for i, c in enumerate((0.95, 0.85, 0.40))]}
    assert [t["conf"] for t in _drawn_tracks(an, min_conf=0.80)] == [0.95, 0.85]
    line = _box_line(an, min_conf=0.80)[0]
    assert "2 of 3 shown (>= 80%)" in line and "min 85%" in line, line
    assert an["people"] == 3  # counts are not filtered
