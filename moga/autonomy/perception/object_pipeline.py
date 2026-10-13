"""Object perception pipeline: LiDAR detection -> LiDAR/radar tracking ->
prediction."""
from __future__ import annotations

import numpy as np

from moga.autonomy.perception.lidar_detection import LidarDetector
from moga.autonomy.prediction.predictor import Predictor
from moga.autonomy.tracking.tracker import MultiObjectTracker
from moga.common.messages import LocalizationEstimate, PredictedObject, SensorBundle, TrackedObject
from moga.map.hd_map import HDMap


class ObjectPerception:
    def __init__(self, hd_map: HDMap, use_radar: bool = True):
        self.detector = LidarDetector(hd_map)
        self.tracker = MultiObjectTracker()
        self.predictor = Predictor(hd_map)
        self.use_radar = use_radar
        self.detections = []
        self.tracks: list[TrackedObject] = []
        self.predictions: list[PredictedObject] = []
        self.last_lidar_t = -1.0
        self.obstacle_points = np.zeros((0, 2))

    def process(self, b: SensorBundle, est: LocalizationEstimate) -> bool:
        """Returns True when the object list was refreshed."""
        updated = False
        if b.lidar is not None:
            self.detections = self.detector.detect(b.lidar, est.x, est.y, est.yaw)
            self.tracker.update_lidar(b.t, self.detections, (est.x, est.y))
            self.obstacle_points = self.detector.last_debug.get("obstacle_world", np.zeros((0, 2)))
            self.last_lidar_t = b.t
            updated = True
        if self.use_radar and b.radar is not None:
            self.tracker.update_radar(b.t, b.radar, (est.x, est.y, est.yaw, est.v, est.yaw_rate))
            updated = True
        if updated:
            self.tracks = self.tracker.output()
            self.predictions = self.predictor.predict(self.tracks)
        return updated
