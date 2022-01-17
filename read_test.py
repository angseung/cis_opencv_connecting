import cv2
from matplotlib import pyplot as plt
from cis_utils import *


def onMouse(event, x, y, flags, param): # 아무스 콜백 함수 구현 ---①
    print(event, x, y, )                # 파라미터 출력
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼 누름인 경우 ---②
        cv2.circle(frame, (x, y), 30, (0, 0, 0), -1) # 지름 30 크기의 검은색 원을 해당 좌표에 그림
        cv2.imshow("frame", frame)          # 그려진 이미지를 다시 표시 ---③


def draw_rectangle(event, x, y, flags, param):
    global x1, y1, click  # 전역변수 사용
    # click = True

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스를 누른 상태
        click = True
        x1, y1 = x, y
        print("사각형의 왼쪽위 설정 : (" + str(x1) + ", " + str(y1) + ")")

    elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
        if click == True:  # 마우스를 누른 상태 일경우
            cv2.rectangle(frame, (x1, y1), (x, y), (255, 0, 0), -1)
            # cv2.circle(img,(x,y),5,(0,255,0),-1)
            print("(" + str(x1) + ", " + str(y1) + "), (" + str(x) + ", " + str(y) + ")")

    elif event == cv2.EVENT_LBUTTONUP:
        click = False;  # 마우스를 때면 상태 변경
        cv2.rectangle(frame, (x1, y1), (x, y), (255, 0, 0), -1)
        # cv2.circle(img,(x,y),5,(0,255,0),-1)

        if x1 < x and y1 < y:
            print("currunt point : (%d, %d) ~ (%d, %d)" % (x1, y1, x, y))
            fig = plt.figure()
            try:
                plt.imshow(frame[x1 : x, y1 : y, :])
                # plt.show()
            except:
                pass


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print(cv2.getBuildInformation())

    ret, frame = cap.read()
    print("Current status :", ret)
    cv2.imshow("frame", frame)
    cv2.setMouseCallback("frame", draw_rectangle)

    # ret, frame = cap.read()

    while True:
        # if not click:
        ret, frame = cap.read()
        # print("Current status :", ret, frame.shape)
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
