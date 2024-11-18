import cv2
import numpy as np
import torch
from torchvision.utils import save_image


def nedi(input, new_dimension, side_2, threshold):
    (m, n) = input.shape
    (N, M) = new_dimension
    img_var = np.zeros((m, n))
    output = cv2.resize(input, new_dimension, interpolation=cv2.INTER_LINEAR)
    for i in range(m):
        for j in range(n):

            dot = input[i, j]
            dot_x1 = max(0, i - side_2)
            dot_x2 = min(i + side_2 + 1, n)
            dot_y1 = max(0, j - side_2)
            dot_y2 = min(j + side_2 + 1, n)
            dot_var = np.var(input[dot_x1:dot_x2, dot_y1:dot_y2])
            img_var[i, j] = dot_var
            if dot_var > threshold:
                output[int(i * M / m - 0.5), int(j * N / n - 0.5)] = dot
    return output, img_var




def NEDI_run(img, new_size, side_2=1, threshold=0.5, ):
    NEDI_img, NEDI_var = nedi(img, new_size, side_2, threshold)
    return NEDI_img