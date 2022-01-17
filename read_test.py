import cv2
from matplotlib import pyplot as plt
from cis_utils import *


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print(cv2.getBuildInformation())

    ret, frame = cap.read()
    cv2.imshow("frame", frame)

    cv2.createTrackbar("ROI Y Position", "frame", 0, 1918, nothing)
    cv2.createTrackbar("ROI X Position", "frame", 0, 1080, nothing)
    cv2.createTrackbar("ROI Width", "frame", 0, 1080, nothing)
    cv2.createTrackbar("ROI Height", "frame", 0, 1918, nothing)

    while True:
        ret, frame = cap.read()
        box = get_roi_box(input_size=(frame.shape[0], frame.shape[1]),
                          pos=(cv2.getTrackbarPos("ROI Y Position", "frame"), cv2.getTrackbarPos("ROI X Position", "frame")),
                          size=(cv2.getTrackbarPos("ROI Height", "frame"), cv2.getTrackbarPos("ROI Width", "frame")),
                          )
        box_mask = box == 255
        frame[box_mask] = 255
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    input = np.rot90(frame[158 : 365, 691 : 826, :])
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
