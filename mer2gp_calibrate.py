"""
MER2GP calibration helper.
Grabs a frame, lets you tweak exposure/gain with track-bars, then saves the chosen settings to camera_profile.json.
Author: Efe Kaya
"""

import json, cv2

PROFILE_JSON = "camera_profile.json"

def save_profile(cam):
    profile = {
        "width":  int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    int(cam.get(cv2.CAP_PROP_FPS)),
        "exposure": cam.get(cv2.CAP_PROP_EXPOSURE),
        "gain":     cam.get(cv2.CAP_PROP_GAIN)
    }
    with open(PROFILE_JSON, "w") as f:
        json.dump(profile, f, indent=2)
    print("Saved profile →", PROFILE_JSON)

def main():
    cam = cv2.VideoCapture(0, cv2.CAP_ANY)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  720)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cam.set(cv2.CAP_PROP_FPS,         302)

    cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Exposure", "Calibrate", 200, 1000,
                       lambda v: cam.set(cv2.CAP_PROP_EXPOSURE, v))
    cv2.createTrackbar("Gain",     "Calibrate",  10,   64,
                       lambda v: cam.set(cv2.CAP_PROP_GAIN,     v))

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        cv2.imshow("Calibrate", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):   # save
            save_profile(cam)
        elif k == 27:       # Esc
            break
    cam.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
