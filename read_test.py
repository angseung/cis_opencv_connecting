import cv2

cap = cv2.VideoCapture(1)
# cap.open(0, cv2.CAP_DSHOW)

if __name__ == "__main__":
    print(cv2.getBuildInformation())

    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()
