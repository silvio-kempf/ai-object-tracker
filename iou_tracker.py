# iou_tracker.py
from typing import Dict, List, Tuple, Optional
import numpy as np

class IOUTracker:
    """
    Einfacher IOU-basierter Multi-Object-Tracker.
    - Greedy Matching nach maximaler IoU pro Detektion
    - Klassen-aware Matching (optional)
    - Tracks verfallen nach 'max_age' Frames ohne Update
    """
    def __init__(self, iou_thr: float = 0.3, max_age: int = 15, class_aware: bool = True):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.class_aware = class_aware

        self.tracks: Dict[int, dict] = {}  # tid -> {bbox, score, cls, age}
        self._next_id = 1

    @staticmethod
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        """
        a,b: [x1,y1,x2,y2]
        """
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        aa = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        ba = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = aa + ba - inter
        return float(inter / union) if union > 0 else 0.0

    def update(self, detections: np.ndarray) -> List[Tuple[int, float, float, float, float, float, int]]:
        """
        detections: shape (N,6): x1,y1,x2,y2,score,cls
        returns: list of (id, x1,y1,x2,y2, score, cls)
        """
        # 1) alle Tracks altern lassen
        for t in self.tracks.values():
            t["age"] += 1
            t["updated"] = False

        # 2) Greedy Zuordnung Detektionen -> existierende Tracks
        for d in detections:
            bb = d[:4]
            score = float(d[4])
            cls = int(d[5])

            best_iou: float = 0.0
            best_id: Optional[int] = None

            for tid, t in self.tracks.items():
                if self.class_aware and t["cls"] != cls:
                    continue
                iou = self.iou(bb, t["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            if best_id is not None and best_iou >= self.iou_thr:
                # Update existierender Track
                t = self.tracks[best_id]
                t["bbox"] = bb
                t["score"] = score
                t["cls"] = cls
                t["age"] = 0
                t["updated"] = True
            else:
                # Neuer Track
                tid = self._next_id; self._next_id += 1
                self.tracks[tid] = {
                    "bbox": bb.astype(float),
                    "score": score,
                    "cls": cls,
                    "age": 0,
                    "updated": True,
                }

        # 3) Stale Tracks entfernen
        stale = [tid for tid, t in self.tracks.items() if t["age"] > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        # 4) Ausgabe sammeln
        out = []
        for tid, t in self.tracks.items():
            x1, y1, x2, y2 = t["bbox"]
            out.append((tid, float(x1), float(y1), float(x2), float(y2), float(t["score"]), int(t["cls"])))
        return out


if __name__ == "__main__":
    # Mini-Tests
    a = np.array([0, 0, 10, 10], dtype=float)
    b = np.array([5, 5, 15, 15], dtype=float)
    print("IoU(a,b) ~ 0.142:", IOUTracker.iou(a, b))

    trk = IOUTracker(iou_thr=0.3, max_age=2, class_aware=True)
    dets = np.array([
        [0, 0, 10, 10, 0.9, 0],   # cls=0
        [50, 50, 80, 80, 0.8, 1], # cls=1
    ])
    print("Update #1:", trk.update(dets))
    # nächster Frame: nur Box 1 leicht verschoben
    dets2 = np.array([
        [1, 1, 11, 11, 0.88, 0],  # sollte Track 1 matchen
    ])
    print("Update #2:", trk.update(dets2))
