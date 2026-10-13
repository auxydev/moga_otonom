"""Library of test maps used by the simulation scenarios."""
from __future__ import annotations

import math

import numpy as np

from moga.common.geodesy import GeoOrigin
from moga.common.geometry import PathBuilder, Polyline
from moga.map.builder import MapBuilder
from moga.map.hd_map import HDMap

# Default geodetic anchor (Ankara).  All maps are expressed in a local ENU frame.
DEFAULT_ORIGIN = GeoOrigin(39.9208, 32.8541, 900.0)


def highway_straight(length: float = 1500.0, lanes: int = 3, speed_limit: float = 25.0,
                     shoulder: float = 3.0) -> HDMap:
    b = MapBuilder("highway_straight", DEFAULT_ORIGIN)
    ref = PathBuilder(0.0, lanes * 3.6, 0.0).straight(length).build()
    b.add_road("hw", ref, n_forward=lanes, lane_width=3.6, speed_limit=speed_limit,
               shoulder_width=shoulder)
    return b.build()


def highway_curvy(lanes: int = 3, speed_limit: float = 25.0) -> HDMap:
    b = MapBuilder("highway_curvy", DEFAULT_ORIGIN)
    ref = (PathBuilder(0.0, lanes * 3.6, 0.0)
           .straight(200).curve(350, 35, 40).straight(150).curve(300, -55, 40)
           .straight(200).curve(400, 25, 40).straight(400).build())
    b.add_road("hw", ref, n_forward=lanes, lane_width=3.6, speed_limit=speed_limit,
               shoulder_width=3.0)
    return b.build()


def rural_two_lane(speed_limit: float = 19.4) -> HDMap:
    b = MapBuilder("rural_two_lane", DEFAULT_ORIGIN)
    ref = (PathBuilder(0.0, 0.0, 0.0)
           .straight(120).curve(90, 60, 25).straight(60).curve(110, -100, 25)
           .straight(80).curve(140, 70, 25).straight(150).build())
    b.add_road("rd", ref, n_forward=1, n_backward=1, lane_width=3.5, speed_limit=speed_limit)
    return b.build()


def loop_track(straight: float = 250.0, radius: float = 80.0, lanes: int = 2,
               speed_limit: float = 19.4) -> HDMap:
    b = MapBuilder("loop_track", DEFAULT_ORIGIN)
    pb = (PathBuilder(0.0, lanes * 3.5, 0.0).straight(straight).curve(radius, 180, 30)
          .straight(straight).curve(radius, 180, 30))
    pts = np.array(pb.pts)
    pts[-1] = pts[0]  # close the loop exactly
    ref = Polyline(pts)
    b.add_road("loop", ref, n_forward=lanes, lane_width=3.5, speed_limit=speed_limit,
               shoulder_width=3.0, loop=True)
    return b.build()


def urban_grid(nx: int = 3, ny: int = 2, block: float = 120.0, lanes: int = 1,
               lane_width: float = 3.5, speed_limit: float = 13.9,
               signals: bool = True, crosswalks: bool = True) -> HDMap:
    """Manhattan-style grid; signalised junctions use split phasing
    (one approach green at a time)."""
    b = MapBuilder(f"urban_grid_{nx}x{ny}", DEFAULT_ORIGIN)
    J = lanes * lane_width + 6.5           # junction half size
    nodes = {(i, j): (i * block, j * block) for i in range(nx) for j in range(ny)}
    # roads: h_i_j from node (i,j) to (i+1,j); v_i_j from (i,j) to (i,j+1)
    for (i, j), (x, y) in nodes.items():
        if (i + 1, j) in nodes:
            ref = Polyline(np.array([[x + J, y], [x + block - J, y]])).resample(0.5)
            b.add_road(f"h{i}{j}", ref, lanes, lanes, lane_width, speed_limit)
        if (i, j + 1) in nodes:
            ref = Polyline(np.array([[x, y + J], [x, y + block - J]])).resample(0.5)
            b.add_road(f"v{i}{j}", ref, lanes, lanes, lane_width, speed_limit)

    for (i, j), (x, y) in nodes.items():
        jid = f"J{i}{j}"
        arms = []  # (name, incoming lanes, outgoing lanes, crosswalk center, road heading)
        if f"h{i}{j}" in b.roads:
            r = b.roads[f"h{i}{j}"]
            arms.append(("E", r.backward, r.forward, (x + J + 3.0, y), 0.0))
        if f"h{i - 1}{j}" in b.roads:
            r = b.roads[f"h{i - 1}{j}"]
            arms.append(("W", r.forward, r.backward, (x - J - 3.0, y), 0.0))
        if f"v{i}{j}" in b.roads:
            r = b.roads[f"v{i}{j}"]
            arms.append(("N", r.backward, r.forward, (x, y + J + 3.0), math.pi / 2))
        if f"v{i}{j - 1}" in b.roads:
            r = b.roads[f"v{i}{j - 1}"]
            arms.append(("S", r.forward, r.backward, (x, y - J - 3.0), math.pi / 2))
        b.add_junction(jid, (x, y), J)
        for a_name, a_in, _, _, _ in arms:
            for b_name, _, b_out, _, _ in arms:
                if a_name == b_name:
                    continue
                for k, lin in enumerate(a_in):
                    # choose matching outgoing lane: keep index (simple lane assignment)
                    lout = b_out[min(k, len(b_out) - 1)]
                    lane = b.connect(lin, lout, jid)
                    # restrict turns for multi-lane approaches
                    if lanes > 1 and ((lane.turn == "left" and k != 0) or
                                      (lane.turn == "right" and k != len(a_in) - 1)):
                        b.map.lanes[lin].successors.remove(lane.id)
                        del b.map.lanes[lane.id]
                        b.map.junctions[jid].lanes.remove(lane.id)
        signalised = signals and len(arms) >= 3
        if signalised:
            for g, (a_name, a_in, _, _, _) in enumerate(arms):
                # far-side signal heads (visible while waiting at the stop line)
                b.add_traffic_light(f"TL_{jid}_{a_name}", list(a_in), group=g, stop_offset=6.5,
                                    ahead=2 * J + 1.0)
        if crosswalks and len(arms) >= 3:
            for a_name, _, _, c, h in arms:
                road_w = 2 * lanes * lane_width + 1.0
                b.add_crosswalk(f"CW_{jid}_{a_name}", c, h + math.pi / 2, road_w, 4.0)
    return b.build()


MAPS = {
    "highway_straight": highway_straight,
    "highway_curvy": highway_curvy,
    "rural_two_lane": rural_two_lane,
    "loop_track": loop_track,
    "urban_grid": urban_grid,
}
