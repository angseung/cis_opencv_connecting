import numpy as np
from matplotlib import pyplot as plt


def nothing(event):
    pass


def get_roi_box(
    input_size: tuple[int, int] = (1080, 1918),
    pos: tuple[int, int] = None,
    size: tuple[int, int] = None,
) -> np.ndarray:

    assert type(input_size) is tuple and type(pos) is tuple and type(size) is tuple

    pos_y, pos_x = pos[0], pos[1]
    height, width = size[0], size[1]
    line_thickness = 5

    # BGR channel image...
    box = np.zeros((*input_size, 3), dtype=np.uint8)
    box[:, :, :] = 255

    box[
        pos_y - line_thickness : pos_y,
        pos_x - line_thickness : pos_x + width + line_thickness,
        :,
    ] = 0
    box[
        pos_y - line_thickness : pos_y + height + line_thickness,
        pos_x - line_thickness : pos_x,
        :,
    ] = 0
    box[
        pos_y + height : pos_y + height + line_thickness,
        pos_x - line_thickness : pos_x + width + line_thickness,
        :,
    ] = 0
    box[
        pos_y - line_thickness : pos_y + height + line_thickness,
        pos_x + width : pos_x + width + line_thickness,
        :,
    ] = 0

    return box


def get_mask_chart(input: np.ndarray = None, MASK_SHOW_OPT: bool = False) -> np.ndarray:
    assert len(input.shape) == 3
    assert input.__class__.__name__ == "ndarray"

    xspace = int(np.round(input.shape[0] / 4))
    yspace = int(np.round(input.shape[1] / 6))

    r = 7
    x_default = -10
    y_default = 3

    mask_chart = np.zeros((input.shape[0], input.shape[1]), dtype=np.uint8)
    xpoint = int(np.round(xspace * 7 / 2) + x_default)
    ypoint = int(np.round(yspace / 2) + y_default)

    for k in range(6):
        mask_chart[
            xpoint - r : xpoint + r, ypoint + yspace * k - r : ypoint + yspace * k + r
        ] = 1

    if MASK_SHOW_OPT:
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(input)

        plt.subplot(122)
        plt.imshow(mask_chart, cmap="gray")

        plt.tight_layout
        plt.show()

    return mask_chart


def get_illuminant(input: np.ndarray = None, mask: np.ndarray = None) -> np.ndarray:
    assert input.__class__.__name__ == "ndarray"
    assert mask.__class__.__name__ == "ndarray"

    if input.shape[0:2] != mask.shape:
        raise ValueError("input and mask must be same size")

    patch_r, patch_g, patch_b = input[:, :, 0], input[:, :, 1], input[:, :, 2]
    patch_r, patch_g, patch_b = patch_r[mask], patch_g[mask], patch_b[mask]

    illuminant = [patch_r.mean(), patch_g.mean(), patch_b.mean()]
    magitude = np.linalg.norm(illuminant)

    return illuminant / magitude


def angular_error(l1: np.ndarray = None, l2: np.ndarray = None):
    ## TODO : find exception case and make exception part...
    assert l1.shape == l2.shape

    l1 = l1 / np.sqrt((l1 ** 2).sum())
    l2 = l2 / np.sqrt((l2 ** 2).sum())

    angle = np.arccos((l1 * l2).sum())

    return np.degrees(angle)


def salt_and_pepper(img, p):
    org_shape = img.shape
    wsize = img.shape[0] * img.shape[1]
    img = img.reshape(wsize)

    thres = 1 - p
    rnd_array = np.random.random(size=wsize)

    img[rnd_array < p] = 0
    img[rnd_array > thres] = 255

    return img.reshape(org_shape)


def hw_RSEPD_fast_HT(input_image=None, Ts=20):
    # start_time = time.time()
    # rowsize = input_image.shape[0]
    # colsize = input_image.shape[1]
    rowsize, colsize = input_image.shape
    # denoised_image = np.zeros((rowsize, colsize), dtype = np.float64)   # TODO: ZEROS_LIKE

    padIm = np.pad(input_image, [1, 1], "symmetric")
    padIm = padIm.astype(np.float64)

    # row_buffer = np.zeros(colsize, )
    MINinW = 0.0
    MAXinW = 255.0

    for i in range(1, rowsize + 1):
        noisyLine = (padIm[i, 1:-1] != MAXinW) & (padIm[i, 1:-1] != MINinW)

        f_hat_line = np.zeros(
            colsize,
        )
        f_bar_line = padIm[i, 1:-1]
        orig_line = padIm[i, 1:-1]

        temp = padIm[i + 1, 1:-1]
        DaLine = abs(padIm[i - 1, 0:-2] - temp)
        DbLine = abs(padIm[i - 1, 1:-1] - temp)
        DcLine = abs(padIm[i - 1, 2:] - temp)

        temp = padIm[i + 1, 1:-1]
        f_hat_DaLine = (padIm[i - 1, 0:-2] + temp) / 2
        f_hat_DbLine = (padIm[i - 1, 1:-1] + temp) / 2
        f_hat_DcLine = (padIm[i - 1, 2:] + temp) / 2

        DLine = np.vstack((DaLine, DbLine, DcLine))  # comparing stack function

        Dmin = np.argmin(DLine, 0)  # min
        zeroIdx = Dmin == 0
        oneIdx = Dmin == 1
        twoIdx = Dmin == 2
        f_hat_line[zeroIdx] = f_hat_DaLine[zeroIdx]
        f_hat_line[oneIdx] = f_hat_DbLine[oneIdx]
        f_hat_line[twoIdx] = f_hat_DcLine[twoIdx]

        noisyRefLine = (padIm[i + 1, 1:-1] == MAXinW) | (padIm[i + 1, 1:-1] == MINinW)
        meanLine = (padIm[i - 1, 0:-2] + 2 * padIm[i - 1, 1:-1] + padIm[i - 1, 2:]) / 4
        f_hat_line[noisyRefLine] = meanLine[noisyRefLine]

        thr = abs(padIm[i, 1:-1] - f_hat_line)
        comp_thr = thr > Ts
        f_bar_line[comp_thr] = f_hat_line[comp_thr]
        f_bar_line[noisyLine] = orig_line[noisyLine]

        row_buffer = np.clip(f_bar_line, 0, 255)
        # denoised_image[i - 1, :] = row_buffer
        padIm[i, :] = np.pad(row_buffer, [1], "symmetric")

    # print("hw_RSEPD_fast_HT end")
    # denoised_image = np.clip(denoised_image, 0, 255)
    # denoised_image = denoised_image.astype(np.uint8)
    # elapsed_time = time.time() - start_time

    # if (ELAPSED_TIME_OPT):
    #     print("Elapsed Time of hw_RSEPD_fast_HT : %d (sec)" %elapsed_time)
    denoised_image = padIm[1:-1, 1:-1].astype(np.uint8)

    return denoised_image


def white_balance(input_image=None, N=4):
    # truncRed = 0
    # truncGreen = 0
    # truncBlue = 0
    input_image = input_image.astype(np.float64)
    output_image = np.zeros(input_image.shape)
    rowsize = (input_image.shape[0]) // N

    now = 0
    gain = np.zeros([input_image.shape[0], input_image.shape[2]], dtype=np.float64)

    for i in range(1, rowsize + 1):
        before = now
        now = np.squeeze(np.sum(input_image[:, N * (i - 1) : N * i, :], 1))

        if i == 1:
            gain = now
        else:
            gain = gain + np.abs(before - now)

    finalGain = np.sum(gain, 0)

    som = np.sqrt((finalGain ** 2).sum())

    gain_R = 1 / (finalGain[0] / som)
    gain_G = 1 / (finalGain[1] / som)
    gain_B = 1 / (finalGain[2] / som)

    output_image[:, :, 0] = gain_R * input_image[:, :, 0]
    output_image[:, :, 1] = gain_G * input_image[:, :, 1]
    output_image[:, :, 2] = gain_B * input_image[:, :, 2]

    output_image = np.clip(output_image, 0, 255.0)
    output_image = np.uint8(output_image)

    return output_image


if __name__ == "__main__":
    a = np.random.randint(low=0, high=255, size=(199, 299, 3), dtype=np.uint8)
    b = get_mask_chart(a, True)
    c = get_illuminant(a, b)
    d = angular_error(np.array([0.5, 0.5, 0.5]), c)

    e = get_roi_box(input_size=(1080, 1918), pos=(225, 255), size=(200, 160))

    plt.imshow(e)
    plt.show()
