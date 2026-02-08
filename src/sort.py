
# Simplified SORT tracker
from filterpy.kalman import KalmanFilter
import numpy as np

class Sort:
    def __init__(self):
        self.track_id = 0
        self.tracks = {}

    def update(self, detections):
        results = []
        for det in detections:
            x1,y1,x2,y2,_ = det
            self.track_id += 1 
            results.append([x1,y1,x2,y2,self.track_id])
        return np.array(results)
