"""
Quick latency benchmark for MER2GP + YOLOv7-tiny ONNX
Run for 5 seconds and print average FPS & 99th-percentile latency
Author: Efe Kaya
"""
import time, statistics, cv2, onnxruntime as ort, numpy as np

MODEL   = "yolov7-tiny.onnx"
SKIP_N  = 4          # 1-in-N frame selection
SECONDS = 5

sess   = ort.InferenceSession(MODEL, providers=["CUDAExecutionProvider"])
iname  = sess.get_inputs()[0].name
cap    = cv2.VideoCapture(0, cv2.CAP_ANY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FPS, 302)

def preprocess(f):
    x = cv2.resize(f, (416,416))[:, :, ::-1].transpose(2,0,1)
    return (x.astype(np.float32)/255.0)[None]

lat = []; frame_id = 0; t0 = time.perf_counter()
while time.perf_counter() - t0 < SECONDS:
    ok, img = cap.read()
    if not ok: continue
    if frame_id % SKIP_N:
        frame_id += 1; continue
    start = time.perf_counter()
    sess.run(None, {iname: preprocess(img)})
    lat.append((time.perf_counter() - start)*1000)  # ms
    frame_id += 1

cap.release()
avg_fps = len(lat) / SECONDS
print(f"Effective FPS: {avg_fps:0.1f}")
print(f"Avg latency:   {statistics.mean(lat):0.2f} ms")
print(f"99-p latency:  {statistics.quantiles(lat, n=100)[-1]:0.2f} ms")
