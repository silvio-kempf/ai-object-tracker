# track_realtime.py
import argparse, cv2, time, numpy as np
from ultralytics import YOLO
from iou_tracker import IOUTracker

COCO = [ 'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant',
'stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon',
'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush' ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--model", type=str, default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou_thr", type=float, default=0.3)
    ap.add_argument("--max_age", type=int, default=15)
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Konnte Quelle nicht öffnen: {args.source}")

    yolo = YOLO(args.model)
    tracker = IOUTracker(iou_thr=args.iou_thr, max_age=args.max_age)

    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        results = yolo(frame[..., ::-1], conf=args.conf, verbose=False)[0]
        dets = []
        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls = int(b.cls[0].item())
            score = float(b.conf[0].item())
            dets.append((x1,y1,x2,y2,score,cls))
        dets = np.array(dets, dtype=float) if dets else np.empty((0,6))

        tracks = tracker.update(dets)

        # draw
        for tid,x1,y1,x2,y2,score,cls in tracks:
            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])
            name = COCO[cls] if 0 <= cls < len(COCO) else str(cls)
            color = (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID {tid} {name} {score:.2f}", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frames += 1
        if frames % 10 == 0:
            now = time.time()
            fps = 10 / (now - t0)
            t0 = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (8,18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("YOLOv8 + IOU Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
