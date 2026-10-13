import numpy as np
import pytest

from moga.map.hd_map import BoundaryType, HDMap, LaneType
from moga.map.library import MAPS, highway_straight, loop_track, urban_grid


@pytest.mark.parametrize("name", list(MAPS))
def test_maps_build(name):
    m = MAPS[name]()
    assert len(m.lanes) > 0
    for lane in m.lanes.values():
        for s in lane.successors:
            assert s in m.lanes
            # successor starts where lane ends
            a = lane.centerline.points[-1]
            b = m.lanes[s].centerline.points[0]
            assert np.hypot(*(a - b)) < 0.05, (lane.id, s)


def test_highway_topology():
    m = highway_straight()
    f0, f1, f2, sh = (m.lanes[k] for k in ("hw_f0", "hw_f1", "hw_f2", "hw_fs"))
    assert f1.left_neighbor == "hw_f0" and f1.right_neighbor == "hw_f2"
    assert f1.can_change_left() and f1.can_change_right()
    assert not f2.can_change_right()            # solid line to the shoulder
    assert f2.right_neighbor == "hw_fs" and sh.lane_type == LaneType.SHOULDER
    assert f0.left_boundary == BoundaryType.SOLID


def test_lane_matching():
    m = highway_straight()
    lane = m.lanes["hw_f1"]
    x, y, yaw, _ = lane.centerline.interpolate(200.0)
    best = m.best_match(x, y + 0.5, yaw)
    assert best.lane_id == "hw_f1"
    assert best.d == pytest.approx(0.5, abs=1e-6)
    assert m.on_road(np.array([[x, y], [x, y + 30]])).tolist() == [True, False]


def test_urban_grid_features():
    m = urban_grid(3, 3)
    assert len(m.traffic_lights) == 4 * 1 + 3 * 4   # centre 4-way + 4 T-junctions x 3
    assert len(m.crosswalks) > 0
    centre = [l for l in m.lanes.values() if l.junction_id == "J11"]
    assert len(centre) == 12                          # 4 arms x 3 turns
    assert any(m.conflicts[l.id] for l in centre)
    turns = {l.turn for l in centre}
    assert turns == {"left", "right", "straight"}


def test_map_json_roundtrip(tmp_path):
    m = urban_grid(2, 2)
    p = tmp_path / "map.json"
    m.save(str(p))
    m2 = HDMap.load(str(p))
    assert set(m2.lanes) == set(m.lanes)
    assert m2.lanes["h00_f0"].successors == m.lanes["h00_f0"].successors


def test_loop_closed():
    m = loop_track()
    lane = m.lanes["loop_f0"]
    assert lane.successors == ["loop_f0"]
    assert np.hypot(*(lane.centerline.points[0] - lane.centerline.points[-1])) < 1e-3
