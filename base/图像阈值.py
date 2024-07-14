import cv2  # openvb 的读取格式是BGR
import numpy as np
import matplotlib.pyplot as plt


# 图像阈值

'''
下面是一些常见的阈值类型：
- `cv2.THRESH_BINARY`：大于阈值的像素值设置为最大值，其他的设置为最小值。
- `cv2.THRESH_BINARY_INV`：大于阈值的像素值设置为最小值，其他的设置为最大值。
- `cv2.THRESH_TRUNC`：大于阈值的像素值设置为阈值，其他的保持不变。
- `cv2.THRESH_TOZERO`：大于阈值的像素值保持不变，其他的设置为最小值。
- `cv2.THRESH_TOZERO_INV`：大于阈值的像素值设置为最小值，其他的保持不变。
'''


def cv2_threshold():
    origin = cv2.imread("./images/girl.jpg", cv2.IMREAD_GRAYSCALE)
    '''
      `cv2.threshold` 是 OpenCV 中用于图像二值化的函数。一个返回值（在大多数情况下为 `ret`）和二值化后的图像（`binary`）。
      1. **`image`**：这是输入图像，通常是一个灰度图像。如果图像是彩色的，需要先将其转换为灰度图像（例如，使用 `cv2.cvtColor` 函数）。
      2. **`127`**：这是阈值。所有像素值高于这个阈值的像素会被设置为最大值（通常为 `255`），而所有像素值低于这个阈值的像素会被设置为最小值（通常为 `0`）。
      3. **`255`**：这是最大值。阈值处理后，高于阈值的像素会被设置为这个值。在 `cv2.THRESH_BINARY` 模式下，这个值是二值化后图像的白色部分。
      4. **`cv2.THRESH_BINARY`**：这是阈值类型。OpenCV 支持多种阈值类型，这里选择的是 `cv2.THRESH_BINARY`，表示二值化模式。如果像素值大于阈值，则设置为最大值；否则，设置为最小值。
    '''

    """
    Overloaded function for applying a threshold to the input image.

    Parameters:
        src: Input image to be thresholded.
        thresh: Threshold value.
        maxval: Maximum value that can be assigned to pixels exceeding the threshold.
        type: Type of thresholding to be applied.
        dst: Optional output image. If not provided, a new image is created.
    Returns:
        A tuple containing the threshold value used and the thresholded image.
    """
    ret, binary = cv2.threshold(origin, 127, 255, cv2.THRESH_BINARY)
    ret, binary_inv = cv2.threshold(origin, 127, 255, cv2.THRESH_BINARY_INV)
    ret, trunc = cv2.threshold(origin, 127, 255, cv2.THRESH_TRUNC)
    ret, tozero = cv2.threshold(origin, 127, 255, cv2.THRESH_TOZERO)
    ret, tozero_inv = cv2.threshold(origin, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['origin', 'binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv']
    images = [origin, binary, binary_inv, trunc, tozero, tozero_inv]

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        # plt.xticks() 函数通常用于设置 X 轴刻度的位置和标签，
        # plt.axis('off') 用于关闭整个轴，包括 X 轴和 Y 轴的刻度、标签、轴线、网格线等。
        plt.xticks([]), plt.yticks([])

    plt.show()


cv2_threshold()
