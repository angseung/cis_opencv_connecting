import time
import datetime
import cv2
from matplotlib import pyplot as plt
from cis_utils import *


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print(cv2.getBuildInformation())

    ret, frame = cap.read()

    if ret:
        cv2.imshow("frame", frame)

    cv2.createTrackbar("ROI X Position", "frame", 0, 1918, nothing)
    cv2.createTrackbar("ROI Y Position", "frame", 0, 1080, nothing)
    cv2.createTrackbar("ROI Width", "frame", 0, 1918, nothing)
    cv2.createTrackbar("ROI Height", "frame", 0, 1080, nothing)

    while True:
        starttime = datetime.datetime.now()
        ret, frame = cap.read()

        if ret:
            pos = (
                cv2.getTrackbarPos("ROI Y Position", "frame"),
                cv2.getTrackbarPos("ROI X Position", "frame"),
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
            box_mask = box == 0
            frame_ori = np.copy(frame)
            frame[box_mask] = 0
            cv2.imshow("frame", frame)
            roi = (pos[1], pos[0], pos[1] + size[1], pos[0] + size[0])

            proceedtime = datetime.datetime.now() - starttime

            if roi[0] and roi[1] and roi[2] and roi[3]:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    input = np.rot90(frame_ori[roi[1] : roi[3], roi[0] : roi[2], :])
                    print(roi)
                    fig = plt.figure()
                    plt.imshow(input)
                    plt.grid(True)
                    plt.show()

                # ground_truth = np.array([0.5, 0.5, 0.5])
                # ground_truth = ground_truth / np.linalg.norm(ground_truth)
                # mask = get_mask_chart(input, False)
                # ill_vec = get_illuminant(input, mask)
                # ill_vec = ill_vec / np.linalg.norm(ill_vec)
                # psnr = cv2.PSNR(input, input, R=255)
                #
                # d = angular_error(ground_truth, ill_vec)
                # print("angular : %.f" % d)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                pass
                # break

    cap.release()
    cv2.destroyAllWindows()

    input = np.rot90(frame_ori[roi[1] : roi[0], roi[3] : roi[2], :])
    fig = plt.figure()
    plt.imshow(input)
    plt.show()

    ground_truth = np.array([0.5, 0.5, 0.5])
    ground_truth = ground_truth / np.linalg.norm(ground_truth)
    mask = get_mask_chart(input, True)
    ill_vec = get_illuminant(input, mask)
    ill_vec = ill_vec / np.linalg.norm(ill_vec)

    d = angular_error(ground_truth, ill_vec)
    print(d)
