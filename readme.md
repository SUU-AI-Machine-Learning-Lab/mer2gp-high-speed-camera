**yolov7-tinydetection-mer2gp-capture.py:
**

How to use

Replace "MODEL_PATH" with the path to your yolov7-tiny.onnx

If you’re using Daheng’s Galaxy SDK, you can still grab frames with OpenCV once the driver exposes the camera as a GenICam/GigE Vision device, otherwise swap cv2.VideoCapture(0) for vendor API calls

Set SKIP_EVERY = 4 to process 1 in N frames (e.g. 1 in 4 ≈ 75 fps effective input to the network)

**mer2gp_calibrate.py:
**

Matches the 4 h calibration block
Records the best known camera settings (resolution, fps, gain, exposure) to a JSON profile that teammates can reload later

**benchmark_latency.py:**

Supports the 9 h pipeline + 11 h YOLO integration blocks
A minimal timer that verifies the capture → inference loop stays below the 3.3 ms/frame budget, with or without “1-in-N” frame skipping
