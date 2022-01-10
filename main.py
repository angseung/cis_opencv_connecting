import sys
import argparse
from typing import Optional, Union
import cv2
import numpy as np
import time
from cis_utils import salt_and_pepper, hw_RSEPD_fast_HT, white_balance

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_device",
        dest="video_device",
        help="Video device # of USB webcam (/dev/video?) [0]",
        default=0,
        type=int,
    )
    arguments = parser.parse_args()
    return arguments


# TODO : find appropriate device path...
def open_usb_capturecard(device: Union[str, int] = 0) -> cv2.VideoCapture:
    return cv2.VideoCapture(
        "nvarguscamerasrc sensor_mode=1 ! "
        "video/x-raw(memory:NVMM), width=(int)400, height=(int)300, format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv ! video/x-raw,format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )


# TODO : find appropriate device path...
def open_camera_device(device: Union[str, int] = 0) -> cv2.VideoCapture:
    return cv2.VideoCapture(device)


def read_cam(video_capture: cv2.VideoCapture = None) -> None:
    if video_capture.isOpened():
        windowName = "CannyDemo"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, 1280, 720)
        cv2.moveWindow(windowName, 0, 0)
        cv2.setWindowTitle(windowName, "Camera Show")
        showWindow = 3  # Show all stages
        showHelp = True
        font = cv2.FONT_HERSHEY_PLAIN
        # helpText = "'Esc' to Quit, '1' for Camera Feed, '2' for Canny Detection, '3' for All Stages. '4' to hide help"
        edgeThreshold = 40
        showFullScreen = False

        while True:
            if (
                cv2.getWindowProperty(windowName, 0) < 0
            ):  # Check to see if the user closed the window
                # This will fail if the user closed the window; Nasties get printed to the console
                break

            (ret_val, frame) = video_capture.read()

            if frame is None:
                continue

            if showWindow == 3:  # Show Camera Display ONLY
                """
                Composite the 2x2 window
                Feed from the camera is RGB, the others gray
                To composite, convert gray images to color."""
                # All images must be of the same type to display in a window
                # frameRs = cv2.resize(frame, (640, 360))
                # hsvRs = cv2.resize(frame1, (640, 360))
                vidBuf = frame
                # blurRs = cv2.resize(blur, (640, 360))

            if showWindow == 1:  # Show NR Test
                start = time.time()

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                noisy_frame = salt_and_pepper(gray_frame, 0.004)
                nrframe = hw_RSEPD_fast_HT(noisy_frame, Ts=20)

                nframeRs = cv2.cvtColor(noisy_frame, cv2.COLOR_GRAY2BGR)
                nrframeRs = cv2.cvtColor(nrframe, cv2.COLOR_GRAY2BGR)

                vidBuf = np.hstack((frame, nframeRs, nrframeRs))
                displayBuf = vidBuf
                end_time = time.time()

            elif showWindow == 2:  # Show AWB Test
                start = time.time()

                wbframe = white_balance(frame)
                vidBuf = np.hstack((frame, wbframe))
                displayBuf = vidBuf
                end_time = time.time()

            elif showWindow == 3:  # Show All Stages
                displayBuf = frame

            if (showHelp == True) and ((showWindow == 1) or (showWindow == 2)):
                fpstxt = "Estimated frames per second : {0}".format(
                    1 / (end_time - start)
                )
                cv2.putText(
                    vidBuf, fpstxt, (11, 20), font, 1.0, (255, 255, 255), 1, cv2.LINE_AA
                )

            cv2.imshow(windowName, displayBuf)

            key = cv2.waitKey(10)

            if key == 27:  # Check for ESC key
                cv2.destroyAllWindows()
                break

            elif key == 49:  # 1 key, show frame
                cv2.setWindowTitle(windowName, "NR TEST")
                showWindow = 1

            elif key == 50:  # 2 key, show Canny
                cv2.setWindowTitle(windowName, "AWB TEST")
                showWindow = 2

            elif key == 51:  # 3 key, show Stages
                cv2.setWindowTitle(windowName, "Camera Show")
                showWindow = 3

    else:
        print("camera open failed")


if __name__ == "__main__":
    arguments = parse_cli_args()
    print("Called with args:")
    print(arguments)
    print("OpenCV version: {}".format(cv2.__version__))
    print("Device Number:", arguments.video_device)
    print(cv2.getBuildInformation())
    if arguments.video_device == 0:
        print("Using on-board camera")
        video_capture = open_usb_capturecard()
    else:
        video_capture = open_camera_device(arguments.video_device)
        # Only do this on external cameras; onboard will cause camera not to read
        video_capture.set(cv2.CAP_PROP_FPS, 30)

    read_cam(video_capture)
    video_capture.release()
    cv2.destroyAllWindows()
