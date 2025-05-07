How to use

Replace "MODEL_PATH" with the path to your yolov7-tiny.onnx.

If you’re using Daheng’s Galaxy SDK, you can still grab frames with OpenCV once the driver exposes the camera as a GenICam/GigE Vision device; otherwise swap cv2.VideoCapture(0) for the vendor API calls.

Set SKIP_EVERY = 4 to process 1 in N frames (e.g. 1 in 4 ≈ 75 fps effective input to the network).
