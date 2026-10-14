# MogaSim senaryo raporu

Oluşturulma: 2026-10-14 12:14:01

**Toplam:** 38 senaryo, **geçen:** 38, **kalan:** 0

| Senaryo | Milestone | Sonuç | Süre [s] | Mesafe [m] | Vmax [km/h] | Hedef | Çarpışma | min TTC [s] | maks. yavaşlama [m/s²] | Şerit RMS [m] | Lok. RMSE [m] | Güvenlik durumları | Modlar | Açıklama |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m0_highway_traffic | M0 | ✅ | 44.18 | 884.18 | 72.24 | ✓ | 0 | ∞ | 0.00 | 0.00 | 0.00 | - | ORACLE |  |
| m0_loop_track | M0 | ✅ | 70.00 | 982.19 | 69.21 | - | 0 | ∞ | 1.80 | 0.25 | 0.00 | - | ORACLE |  |
| m0_urban_lights | M0 | ✅ | 60.00 | 156.66 | 38.58 | - | 0 | ∞ | 1.96 | 0.00 | 0.00 | - | ORACLE |  |
| m1_gps_loop_rtk | M1 | ✅ | 96.36 | 1227.87 | 72.46 | ✓ | 0 | ∞ | 3.48 | 0.10 | 0.02 | NOMINAL | GOAL_REACHED, LANE_CHANGE, LANE_KEEP, PREPARE_LC |  |
| m1_rural_rtk | M1 | ✅ | 71.42 | 905.67 | 64.93 | ✓ | 0 | ∞ | 2.24 | 0.06 | 0.02 | NOMINAL | GOAL_REACHED, LANE_KEEP |  |
| m1_urban_route | M1 | ✅ | 96.90 | 593.09 | 45.33 | ✓ | 0 | ∞ | 2.27 | 0.17 | 0.01 | NOMINAL | GOAL_REACHED, LANE_KEEP |  |
| m1_highway_tunnel | M1 | ✅ | 62.28 | 1246.63 | 91.38 | ✓ | 0 | ∞ | 3.11 | 0.01 | 0.02 | NOMINAL → DEGRADED → NOMINAL | GOAL_REACHED, LANE_KEEP |  |
| m1_rural_multipath | M1 | ✅ | 71.18 | 906.23 | 65.13 | ✓ | 0 | ∞ | 2.32 | 0.06 | 0.29 | NOMINAL | GOAL_REACHED, LANE_KEEP |  |
| m2_curvy_sps_lane_keep | M2 | ✅ | 78.82 | 1684.92 | 92.12 | ✓ | 0 | ∞ | 2.94 | 0.05 | 0.88 | NOMINAL | GOAL_REACHED, LANE_KEEP |  |
| m2_rural_dgps_fusion | M2 | ✅ | 70.98 | 906.39 | 65.58 | ✓ | 0 | ∞ | 2.24 | 0.07 | 0.42 | NOMINAL | GOAL_REACHED, LANE_KEEP |  |
| m2_highway_sps_lane_change | M2 | ✅ | 53.10 | 1067.83 | 91.21 | ✓ | 0 | ∞ | 2.97 | 0.20 | 0.99 | NOMINAL | GOAL_REACHED, LANE_CHANGE, LANE_KEEP, PREPARE_LC |  |
| m2_curvy_long_tunnel | M2 | ✅ | 82.52 | 1683.49 | 91.38 | ✓ | 0 | ∞ | 2.98 | 0.03 | 0.14 | NOMINAL → DEGRADED → NOMINAL | GOAL_REACHED, LANE_KEEP |  |
| m2_rain_rural | M2 | ✅ | 71.54 | 905.77 | 64.27 | ✓ | 0 | ∞ | 2.26 | 0.04 | 0.22 | DEGRADED | GOAL_REACHED, LANE_KEEP, LOCALIZING |  |
| m3_highway_perception | M3 | ✅ | 62.38 | 1256.56 | 90.98 | ✓ | 0 | 50.01 | 2.42 | 0.10 | 0.01 | NOMINAL | GOAL_REACHED, LANE_KEEP |  |
| m3_acc_stop_and_go | M3 | ✅ | 62.66 | 1306.81 | 90.81 | ✓ | 0 | 28.49 | 2.97 | 0.18 | 0.01 | NOMINAL | GOAL_REACHED, LANE_CHANGE, LANE_KEEP, PREPARE_LC |  |
| m3_lead_hard_brake | M3 | ✅ | 35.00 | 645.73 | 92.94 | - | 0 | 4.41 | 8.56 | 0.27 | 0.01 | NOMINAL | EMERGENCY_STOP, LANE_CHANGE, LANE_KEEP, PREPARE_LC |  |
| m3_stopped_vehicle | M3 | ✅ | 45.00 | 489.22 | 54.49 | - | 0 | 3.84 | 2.83 | 0.93 | 0.02 | NOMINAL | BLOCKED, BYPASS, LANE_KEEP |  |
| m3_pedestrian_jaywalk | M3 | ✅ | 45.52 | 442.59 | 57.92 | ✓ | 0 | 2.05 | 7.28 | 0.06 | 0.02 | NOMINAL | EMERGENCY_STOP, GOAL_REACHED, LANE_KEEP |  |
| m4_overtake_slow_truck | M4 | ✅ | 87.46 | 1756.76 | 90.68 | ✓ | 0 | 7.23 | 2.82 | 0.18 | 0.01 | NOMINAL | GOAL_REACHED, LANE_CHANGE, LANE_KEEP, PREPARE_LC |  |
| m4_cut_in | M4 | ✅ | 59.74 | 1246.59 | 90.28 | ✓ | 0 | ∞ | 3.00 | 0.01 | 0.01 | NOMINAL | GOAL_REACHED, LANE_KEEP |  |
| m4_static_obstacle_lane_change | M4 | ✅ | 66.04 | 1157.14 | 92.94 | ✓ | 0 | 2.53 | 4.95 | 0.17 | 0.01 | NOMINAL | GOAL_REACHED, LANE_CHANGE, LANE_KEEP, PREPARE_LC |  |
| m4_route_lane_changes_traffic | M4 | ✅ | 60.56 | 1216.08 | 91.24 | ✓ | 0 | 20.66 | 2.30 | 0.20 | 0.01 | NOMINAL | LANE_CHANGE, LANE_KEEP, PREPARE_LC |  |
| m4_rural_bypass | M4 | ✅ | 56.76 | 357.03 | 63.06 | ✓ | 0 | 1.86 | 2.15 | 0.76 | 0.02 | NOMINAL | BLOCKED, BYPASS, GOAL_REACHED, LANE_KEEP |  |
| m4_aeb_occluded_pedestrian | M4 | ✅ | 44.50 | 357.45 | 46.05 | ✓ | 0 | 0.54 | 8.17 | 0.10 | 0.02 | NOMINAL | AEB, EMERGENCY_STOP, GOAL_REACHED, LANE_KEEP |  |
| m5_red_light_stop_go | M5 | ✅ | 37.26 | 157.62 | 38.74 | ✓ | 0 | ∞ | 1.81 | 0.01 | 0.01 | NOMINAL | GOAL_REACHED, LANE_KEEP, WAIT_LIGHT |  |
| m5_yellow_dilemma | M5 | ✅ | 43.42 | 277.55 | 49.53 | ✓ | 0 | ∞ | 3.25 | 0.01 | 0.01 | NOMINAL | GOAL_REACHED, LANE_KEEP, WAIT_LIGHT |  |
| m5_crosswalk_pedestrians | M5 | ✅ | 40.40 | 157.48 | 37.46 | ✓ | 0 | 3.71 | 1.77 | 0.04 | 0.01 | NOMINAL | GOAL_REACHED, LANE_KEEP, YIELD_PED |  |
| m5_unsignalized_junction | M5 | ✅ | 28.78 | 157.43 | 36.00 | ✓ | 0 | 4.82 | 1.77 | 0.01 | 0.02 | NOMINAL | GOAL_REACHED, LANE_KEEP, YIELD_JUNCTION |  |
| m5_queue_at_red | M5 | ✅ | 44.20 | 172.53 | 37.51 | ✓ | 0 | 6.48 | 1.87 | 0.03 | 0.02 | NOMINAL | GOAL_REACHED, LANE_KEEP, WAIT_LIGHT, YIELD_JUNCTION |  |
| m5_urban_mission | M5 | ✅ | 177.56 | 627.12 | 40.79 | ✓ | 0 | 3.30 | 3.90 | 0.49 | 0.01 | NOMINAL | BLOCKED, BYPASS, GOAL_REACHED, LANE_KEEP, WAIT_LIGHT, YIELD_PED |  |
| m6_lidar_failure | M6 | ✅ | 50.00 | 420.35 | 88.55 | - | 0 | 30.11 | 3.51 | 0.01 | 0.01 | NOMINAL → MRM → MRC | LANE_KEEP, MRC, MRM_STOP |  |
| m6_camera_failure_pullover | M6 | ✅ | 70.00 | 382.04 | 88.55 | - | 0 | 30.11 | 3.46 | 0.02 | 0.01 | NOMINAL → MRM → MRC | LANE_KEEP, MRC, MRM_PULLOVER |  |
| m6_radar_failure_degraded | M6 | ✅ | 81.98 | 1636.31 | 88.55 | ✓ | 0 | 30.11 | 2.68 | 0.01 | 0.01 | NOMINAL → DEGRADED | LANE_KEEP |  |
| m6_gnss_outage_recovery | M6 | ✅ | 65.60 | 890.27 | 65.74 | ✓ | 0 | ∞ | 2.27 | 0.09 | 0.18 | NOMINAL → DEGRADED → NOMINAL | GOAL_REACHED, LANE_KEEP |  |
| m6_steering_degraded | M6 | ✅ | 60.00 | 164.42 | 56.21 | - | 0 | ∞ | 3.34 | 0.06 | 0.01 | NOMINAL → MRM → MRC | LANE_KEEP, MRC, MRM_PULLOVER |  |
| m6_heavy_rain_odd_exit | M6 | ✅ | 90.00 | 729.76 | 90.04 | - | 0 | ∞ | 2.76 | 0.01 | 0.01 | NOMINAL → DEGRADED → MRM → MRC | LANE_KEEP, MRC, MRM_PULLOVER |  |
| m6_urban_light_detector_failure | M6 | ✅ | 40.00 | 50.39 | 36.00 | - | 0 | ∞ | 3.25 | 0.01 | 0.01 | NOMINAL → MRM → MRC | LANE_KEEP, MRC, MRM_PULLOVER |  |
| m6_lidar_degraded | M6 | ✅ | 42.44 | 172.41 | 31.97 | ✓ | 0 | 5.92 | 1.71 | 0.01 | 0.02 | NOMINAL → DEGRADED | GOAL_REACHED, LANE_KEEP, WAIT_LIGHT |  |

## Senaryo açıklamaları

* **m0_highway_traffic** (M0) – Oracle drives on a 3-lane highway with IDM traffic and a cut-in.
* **m0_loop_track** (M0) – Oracle laps the oval track at the speed limit.
* **m0_urban_lights** (M0) – Oracle drives straight through signalised junctions with traffic.
* **m1_gps_loop_rtk** (M1) – Multi-waypoint GNSS mission around the oval incl. a lane change; RTK GNSS.
* **m1_rural_rtk** (M1) – Winding two-lane rural road to a GNSS destination; RTK GNSS.
* **m1_urban_route** (M1) – Lane-level routing through junctions (turns) to two GNSS waypoints; lights forced green (light perception comes in M5).
* **m1_highway_tunnel** (M1) – 350 m tunnel without GNSS at 90 km/h: IMU + odometry dead reckoning.
* **m1_rural_multipath** (M1) – RTK-float GNSS with 5% multipath outliers (8 m): the EKF must gate them.
* **m2_curvy_sps_lane_keep** (M2) – Curvy highway at 90 km/h with standalone GNSS (~1.5 m error): camera lane fusion keeps the vehicle centred.
* **m2_rural_dgps_fusion** (M2) – Winding two-lane road with DGPS + multipath: lane fusion removes the lateral GNSS bias (cf. M1).
* **m2_highway_sps_lane_change** (M2) – Standalone GNSS; GNSS goal two lanes to the right: lane identity from marking types, association through the lane changes.
* **m2_curvy_long_tunnel** (M2) – ~800 m of curves without GNSS: camera lane updates bound the lateral error.
* **m2_rain_rural** (M2) – Heavy rain (low contrast, noise) with DGPS on the rural road.
* **m3_highway_perception** (M3) – Dense 3-lane traffic (cars, trucks): detection / tracking quality and ACC following in the ego lane.
* **m3_acc_stop_and_go** (M3) – Lead vehicle slows down, stops and drives off again (stop-and-go ACC).
* **m3_lead_hard_brake** (M3) – Lead vehicle performs an emergency stop (6 m/s^2) at 90 km/h.
* **m3_stopped_vehicle** (M3) – Broken-down car in the lane behind a curve on a two-lane road with a solid centre line: stop safely behind it (from M4 on: pass it through the free opposite lane with sufficient clearance).
* **m3_pedestrian_jaywalk** (M3) – A pedestrian crosses the road away from a crosswalk when the vehicle is 45 m away: yield, then continue to the goal.
* **m4_overtake_slow_truck** (M4) – Slow truck in the right lane, traffic in the middle lane: change left through a gap, overtake and return to the right lane (keep-right rule).
* **m4_cut_in** (M4) – A faster car cuts in ~12 m ahead and then slows down: the vehicle opens the gap smoothly without emergency.
* **m4_static_obstacle_lane_change** (M4) – Broken-down car in the ego lane with traffic on both neighbour lanes: merge into a free gap and pass it.
* **m4_route_lane_changes_traffic** (M4) – GNSS goal in the right lane while the right and middle lanes are busy: two mandatory lane changes into gaps.
* **m4_rural_bypass** (M4) – Broken-down vehicle on a two-lane road with oncoming traffic: wait, then bypass using the opposite lane when it is free.
* **m4_aeb_occluded_pedestrian** (M4) – A pedestrian runs out from behind a parked van 17 m ahead at 40 km/h: the planner and the independent AEB layer must prevent a collision.
* **m5_red_light_stop_go** (M5) – Signalised junction (split phasing, fixed time): the light is red on arrival – perceive it, stop at the stop line, go on green.
* **m5_yellow_dilemma** (M5) – Two yellow lights: close to the first stop line the vehicle continues (stopping would need hard braking), far from the second it stops.
* **m5_crosswalk_pedestrians** (M5) – Green light, but pedestrians cross the crosswalks before and behind the junction: yield (without blocking the junction), then continue.
* **m5_unsignalized_junction** (M5) – Unsignalised 4-way junction, two vehicles approach on the crossing road at the same time: yield, then cross through the gap behind them.
* **m5_queue_at_red** (M5) – Two cars ahead stop at a red light: queue behind them (they are not obstacles to bypass) and follow them on green.
* **m5_urban_mission** (M5) – GNSS mission across a signalised city grid: left turn behind a lead car, 4-way junction with cross traffic, double-parked van with oncoming traffic, pedestrians on crosswalks, right turns, unsignalised corner.
* **m6_lidar_failure** (M6) – LiDAR fails at 90 km/h in traffic: no safe lane change possible any more -> in-lane stop (radar keeps track of the traffic), hazard lights, MRC.
* **m6_camera_failure_pullover** (M6) – Camera failure (loss of redundancy) in the middle lane: pull over through the traffic to the hard shoulder and stop there (MRC).
* **m6_radar_failure_degraded** (M6) – Radar failure: loss of redundancy only – continue in degraded mode (80 km/h, larger time gap, no discretionary lane changes) to the goal.
* **m6_gnss_outage_recovery** (M6) – 30 s GNSS receiver failure on a winding road: dead reckoning + camera lane localisation in degraded mode, back to nominal when GNSS returns.
* **m6_steering_degraded** (M6) – Steering actuator loses 45 % of its effectiveness on a rural road: the response mismatch is detected, the vehicle pulls to the right edge and stops.
* **m6_heavy_rain_odd_exit** (M6) – Rain sets in (degraded: 80 km/h) and turns into a cloudburst beyond the ODD: pull over to the shoulder and stop.
* **m6_urban_light_detector_failure** (M6) – Traffic light detector fails in the city: signals ahead cannot be read -> pull over to the curb before the junction (not on the crosswalk) and stop.
* **m6_lidar_degraded** (M6) – Dirty LiDAR cover (half of the returns lost): degraded mode, the trip is completed with restrictions.
