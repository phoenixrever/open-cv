import cv2  # openvb 的读取格式是BGR
import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片


# 图像平滑

'''
下面是一些常见的图像平滑类型：
cv2.blur:均值滤波（Mean Filtering） 使用简单的平均卷积特效进行平滑。简单来说 如果图像有像素点 就会减少
cv2.boxFilter:方框滤波（Box Filter） 类似于均值滤波，但使用的是归一化的方框滤波。
cv2.medianBlur:中值滤波（Median Blurring）  使用中值滤波器进行平滑，特别适用于去除椒盐噪声。
cv2.GaussianBlur:高斯滤波（Gaussian Blurring） 使用高斯卷积核对图像进行平滑，能够更好地保留边缘信息。
cv2.bilateralFilter:双边滤波（Bilateral Filtering） 双边滤波既能平滑图像又能保留边缘信息，是一种非线性滤波器。
cv2.adaptiveBilateralFilter(删除了此函数): 自适应均值滤波（Adaptive Mean Filtering） 自适应均值滤波根据图像的局部变化自适应地调整滤波窗口的大小。
cv2.fastNlMeansDenoisingColored : 非局部均值滤波（Non-Local Means Denoising） 非局部均值滤波用于去除噪声，能够更好地保留图像细节。

cv2.Sobel :Sobel 滤波（Sobel Filtering）虽然主要用于边缘检测，但也可以用于图像平滑和增强。
          Sobel 算子在 OpenCV 中通常用于边缘检测，它返回的结果是灰度图像，而不是彩色图像。
          这是因为 Sobel 算子在计算梯度时，会将图像转换为灰度图像处理，从而得到每个像素点的梯度强度和方向。
'''


def cv2_smooth():
    image = cv2.imread("./images/girl.jpg")
    """
      cv2.blur(src, ksize[, dst[, anchor[, borderType]]]) -> dst

        src: 输入图像，可以是任意通道数的单精度浮点型或者整数型图像。

        ksize: 平滑核的大小。这是一个二元元组 (width, height)，指定了核的宽度和高度。核的大小必须是正奇数，比如 (3, 3)、(3, 3) 等。

        dst: 输出与输入图像相同大小和类型的图像。如果没有提供，则函数会修改输入图像。

        anchor: 核的锚点。它指示了核中心相对于输入图像中的哪个位置。默认值为 (-1, -1)，表示核中心位于核的中心。

        borderType: 边界模式。控制在处理图像边界时的行为。默认值为 cv2.BORDER_DEFAULT。

      返回值
      
        dst: 函数不会直接修改原始图像，而是将处理后的图像存储在指定的 dst 参数中,如果没有提供 dst 参数，则会创建一个新的数组来存储处理后的图像，并返回这个数组。
    """
    blur = cv2.blur(image, (3, 3))
    # normalize True 归一化（求平均 也就是/9） 和均值滤波一样 , false  就是 不求kernel的平均值 而是直接用9个格子的和 超出255的值取255
    box_filter = cv2.boxFilter(image, -1, (3, 3), normalize=False)
    median_blur = cv2.medianBlur(image, 5)
    bilateral_blur = cv2.bilateralFilter(image, 9, 75, 75)
    gaussian_blur = cv2.GaussianBlur(image, (3, 3), 0)
    nl_means_denoising = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_blur = cv2.sqrt(cv2.addWeighted(sobelx**2, 0.5, sobely**2, 0.5, 0))

    titles = ['image', 'blur', 'box_filter', 'median_blur', 'bilateral_blur',
              'gaussian_blur', 'nl_means_denoising', 'sobel_blur']

    images = [image, blur, box_filter, median_blur, bilateral_blur,
              gaussian_blur, nl_means_denoising, sobel_blur]

    # 太小看不清

    # for i in range(8):
    #   # OpenCV 读取图像是 BGR 通道顺序，需要转换为 RGB
    #     if i < 7:
    #         image_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    #         plt.subplot(3, 3, i+1), plt.imshow(image_rgb)
    #     else:
    #         plt.subplot(3, 3, i+1), plt.imshow(images[i])
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()

    image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
    blur = cv2.resize(blur, (int(blur.shape[1] / 4), int(blur.shape[0] / 4)))
    box_filter = cv2.resize(box_filter, (int(box_filter.shape[1] / 4), int(box_filter.shape[0] / 4)))
    median_blur = cv2.resize(median_blur, (int(median_blur.shape[1] / 4), int(median_blur.shape[0] / 4)))
    bilateral_blur = cv2.resize(bilateral_blur, (int(bilateral_blur.shape[1] / 4), int(bilateral_blur.shape[0] / 4)))
    gaussian_blur = cv2.resize(gaussian_blur, (int(gaussian_blur.shape[1] / 4), int(gaussian_blur.shape[0] / 4)))
    nl_means_denoising = cv2.resize(
        nl_means_denoising, (int(nl_means_denoising.shape[1] / 4),
                             int(nl_means_denoising.shape[0] / 4)))
    # Sobel 灰度图不能放里面
    res = np.hstack((image, blur, box_filter, median_blur, bilateral_blur,
                     gaussian_blur, nl_means_denoising))

    cv2.imshow("图像平滑", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cv2_smooth()

# 补充 均值滤波 的原理

"""
    核的定义：
      cv2.blur 函数接受一个核（kernel）作为参数，这个核是一个矩阵，用来定义滤波器的形状和大小。核的大小由参数 ksize 定义，通常是一个二元元组 (width, height)，表示核的宽度和高度。
    
    图像遍历：
      对于输入的图像，从左上角开始，将核放置在每个像素位置上。核的中心与当前像素对齐。
    
    平均值计算：
      对于每个像素位置，核覆盖的区域内的所有像素值取平均值。
      平均值计算是基于核中所有像素的权重，通常情况下，这些权重都是相同的。
    
    
    边界处理：
      在核覆盖图像边界的情况下，需要对边界像素进行填充处理。OpenCV 提供了不同的边界处理方式，例如复制边界像素、反射边界像素等。
      
    结果生成：
      处理后的结果图像中，每个像素的值都是根据核在原始图像上计算得出的平均值。
      
      示例说明：
        假设我们有以下 3x3 的核：
        1/9  1/9  1/9
        1/9  1/9  1/9
        1/9  1/9  1/9
        
        对于一个简单的图像区域，如下所示：
        [ 10, 20, 30 ]
        [ 40, 50, 60 ]
        [ 70, 80, 90 ]
        
        应用 3x3 的均值滤波核，例如上述的核，计算方法如下：
        对于中心像素 [50]，计算周围 3x3 区域的平均值：(10 + 20 + 30 + 40 + 50 + 60 + 70 + 80 + 90) / 9 = 50。
        这样，中心像素 [50] 的值被平均化为 50，即得到平滑后的效果。
        
        
        边界填充方式
          1. **默认边界填充方式 (`cv2.BORDER_DEFAULT`)**：
            - 默认使用的边界填充方式，根据图像边界的像素值进行填充。

          2. **常量填充方式 (`cv2.BORDER_CONSTANT`)**：
            - 使用指定的常数填充边界，可以通过 `value` 参数指定填充的像素值。

          3. **复制边界像素填充方式 (`cv2.BORDER_REPLICATE`)**：
            - 复制图像边界的最后一个像素。

          4. **反射边界像素填充方式 (`cv2.BORDER_REFLECT`)**：
            - 对边界像素进行镜像反射。

          5. **边界像素反射加复制填充方式 (`cv2.BORDER_WRAP`)**：
            - 使用镜像反射的方式填充。
            
        举例
          [ 5, 10, 20 ]
          [ 10, 10, 20 ]
          [ 40, 40, 50 ]

        左上角元素  填充取最近元素的值
          [ 5, 5, 10 ]
          [ 5, 5, 10 ]
          [ 10, 10, 10 ]

        计算这些像素的平均值：(5 + 5 + 10 + 5 + 5 + 10 + 10 + 10 + 10) / 9 = 7.78（保留两位小数）。
        因此，左上角像素 [5] 在应用 3x3 的均值滤波核后，得到的平均值约为 7.78。

      
    """
