import time
import numpy as np
import cv2
from cis_utils import (
    nothing,
    get_roi_box,
    get_illuminant,
    get_mask_chart,
    angular_error,
    get_psnr,
)
"""
cam_num == 0 if there is no built-in webcam
else cam_num == 1
"""

# cam_num = 0
cam_num = 1

if __name__ == "__main__":
    ROI_SET = False
    cap = cv2.VideoCapture(cam_num)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1918)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print(cv2.getBuildInformation())
    font = cv2.FONT_HERSHEY_PLAIN

    ret, frame = cap.read()
    cv2.namedWindow("frame")
    cv2.namedWindow("control")
    cv2.resizeWindow(winname='frame', width=1280, height=720)
    cv2.resizeWindow(winname='control', width=960, height=180)

    if ret:
        cv2.imshow("frame", frame)

    cv2.createTrackbar("ROI X", "control", 0, 1918 // 2, nothing)

    cv2.createTrackbar("ROI Y", "control", 0, 1080 // 2, nothing)

    cv2.createTrackbar("ROI Width", "control", 0, 1918 // 2, nothing)

    cv2.createTrackbar("ROI Height", "control", 0, 1080 // 2, nothing)

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if ret:
            pos = (
                cv2.getTrackbarPos("ROI Y", "control"),
                cv2.getTrackbarPos("ROI X", "control"),
            )
            size = (
                cv2.getTrackbarPos("ROI Height", "control"),
                cv2.getTrackbarPos("ROI Width", "control"),
            )
            box = get_roi_box(
                input_size=(frame.shape[0], frame.shape[1]),
                pos=pos,
                size=size,
            )
            frame_ori = np.copy(frame)
            frame[box == 0] = 0
            roi = (pos[1], pos[0], pos[1] + size[1], pos[0] + size[0])

            if roi[0] and roi[1] and roi[2] and roi[3]:
                # when ROI is set
                if cv2.waitKey(1) & 0xFF in [ord("D"), ord("d")]:
                    ROI_SET = True

                if ROI_SET:
                    input = np.rot90(frame_ori[roi[1] : roi[3], roi[0] : roi[2], :])
                    denoised = frame_ori[1:540, 1918 // 2 + 1 : 1918, :]
                    origin = frame_ori[540 : 1080 - 1, 0 : 1918 // 2 - 1, :]
                    psnr = get_psnr(origin, denoised, max_val=255.0)

                    # AWB angular error performance
                    ground_truth = np.array([0.5, 0.5, 0.5])
                    ground_truth = ground_truth / np.linalg.norm(ground_truth)
                    mask = get_mask_chart(input, False)
                    ill_vec = get_illuminant(input, mask)
                    ill_vec = ill_vec / np.linalg.norm(ill_vec)
                    angle = angular_error(ground_truth, ill_vec)
                    performance = (
                        "RSEPD PSNR : %.2f(dB), AWB Angular Error : %.2f(degree)"
                        % (psnr, angle)
                    )
                    cv2.putText(
                        frame,
                        performance,
                        (20, 60),
                        color=(0, 0, 0),
                        fontFace=font,
                        fontScale=1.5,
                        thickness=2,
                    )

            end_time = time.time()
            fpstxt = "Estimated frames per second : %.2f" % (
                1 / (end_time - start_time)
            )
            cv2.putText(
                frame,
                fpstxt,
                (20, 20),
                color=(0, 0, 0),
                fontFace=font,
                fontScale=1.5,
                thickness=2,
            )
            frame = cv2.resize(frame, dsize=(1280, 720))
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF in [ord("Q"), ord("q")]:
                break

    cap.release()
    cv2.destroyAllWindows()
