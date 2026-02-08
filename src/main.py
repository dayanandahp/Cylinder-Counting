
import cv2
import argparse
from ultralytics import YOLO
from tracker import Sort

def main(video_path):
    model = YOLO("yolov8n.pt")
    tracker = Sort()
    cap = cv2.VideoCapture(video_path)

    counted_ids = set()
    total_count = 0
    line_y = 300  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4, classes=None)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            detections.append([x1, y1, x2, y2, conf])

        tracks = tracker.update(detections)

        for trk in tracks:
            x1, y1, x2, y2, track_id = trk.astype(int)
            cy = (y1 + y2) // 2

            if cy > line_y and track_id not in counted_ids:
                counted_ids.add(track_id)
                total_count += 1

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f'ID:{track_id}',(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(0,0,255),2)
        cv2.putText(frame,f'Total Count: {total_count}',(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        cv2.imshow("Cylinder Counter", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("Final Cylinder Count:", total_count)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    main(args.video)
