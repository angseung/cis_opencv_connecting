import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
from cis_utils import (
    nothing,
    get_roi_box,
    get_illuminant,
    get_mask_chart,
    angular_error,
    get_psnr,
)


if __name__ == "__main__":
    ROI_SET = False
    cap = cv2.VideoCapture(0)
    print(cv2.getBuildInformation())
    font = cv2.FONT_HERSHEY_PLAIN

    ret, frame = cap.read()
    cv2.namedWindow("frame")

    if ret:
        cv2.imshow("frame", frame)

    cv2.createTrackbar("ROI X", "frame", 0, 1918 // 2, nothing)
    cv2.createTrackbar("ROI Y", "frame", 0, 1080 // 2, nothing)
    cv2.createTrackbar("ROI Width", "frame", 0, 1918 // 4, nothing)
    cv2.createTrackbar("ROI Height", "frame", 0, 1080 // 4, nothing)

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if ret:
            pos = (
                cv2.getTrackbarPos("ROI Y", "frame"),
                cv2.getTrackbarPos("ROI X", "frame"),
            )
            size = (
                cv2.getTrackbarPos("ROI Height", "frame"),
                cv2.getTrackbarPos("ROI Width", "frame"),
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
                    # TODO : Validate with revised logic...
                    denoised = frame_ori[0:540, 1918 // 2 : 1918, :]
                    origin = frame_ori[540:1080, 0 : 1918 // 2, :]
                    psnr = get_psnr(origin, denoised, max_val=255)

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
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF in [ord("Q"), ord("q")]:
                break

    cap.release()
    cv2.destroyAllWindows()
