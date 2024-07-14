import cv2  # openvb 的读取格式是BGR
import numpy as np
import matplotlib.pyplot as plt


def cv2_imshow(name, image):
    # 参数1表示以彩色模式读取图像。如果是0 (IMREAD_GRAYSCALE)，则以灰度模式读取。
    image = cv2.imread(image, cv2.IMREAD_COLOR)

    # 图像的形状：450行（高度），450列（宽度），3个通道（颜色通道：红、绿、蓝）
    print("image shape: ", image.shape)

    print("image type: ", image.dtype)  # 图像的数据类型：uint8（每个像素值是一个8位无符号整数(0~255)）

    print("image size: ", image.size)  # 图像的总像素数：450 * 450 * 3 = 607500

    print("image ndim: ", image.ndim)  # 图像的维度数：3（行、列和通道）

    print("image itemsize: ", image.itemsize)  # 每个像素的字节大小：1字节（对于uint8类型）

    print("image flags: ", image.flags)  # 图像的内存布局信息：
    # C_CONTIGUOUS : True  # 数组在内存中是行优先（C顺序）的连续存储
    # F_CONTIGUOUS : False  # 数组在内存中不是列优先（Fortran顺序）的连续存储
    # OWNDATA : True  # 数组拥有它所使用的内存
    # WRITEABLE : True  # 数组是可写的
    # ALIGNED : True  # 数组数据按适当的字节边界对齐
    # WRITEBACKIFCOPY : False  # 数组没有备份写入的复制

    area = image[0:100, 0:200]  # 截取图像的区域    0:100 表示从0行到100行  0:200 表示从0列到200列

    b, g, r = cv2.split(image)  # 通道分离
    print("b", b)

    image_merge = cv2.merge([b, g, r])  # 通道合并

    # 只保留R(RED)通道 image shape:  (450, 450, 3) blue green red
    # copyR = image.copy()
    # copyR[:,:,0] = 0
    # copyR[:,:,1] = 0
    # cv2.imshow(name, copyR)

    # 只保留g通道
    # copyR = image.copy()
    # copyR[:,:,0] = 0
    # copyR[:,:,2] = 0
    # cv2.imshow(name, copyR)

    # 只保留b通道
    # copyR = image.copy()
    # copyR[:,:,1] = 0
    # copyR[:,:,1] = 0
    # cv2.imshow(name, copyR)

    cv2.imshow(name, image)
    cv2.waitKey(0)  # 等待键盘事件 毫秒级  参数0表示无限期等待， 3000表示图片显示3s 直到按下任意键。
    cv2.destroyAllWindows()  # 这行代码使用OpenCV的cv2.destroyAllWindows函数关闭所有OpenCV窗口。


# cv2_imshow("image", "./images/moon.jpg")

def cv2_videoShow(name, video):
    video = cv2.VideoCapture(video)

    if video.isOpened():
        opened = True
    else:
        opened = False
        print("视频打开失败")

    while opened:
        # 从视频捕获对象 'video' 中读取一帧图像，返回值 'ret' 表示读取是否成功，'frame' 是读取到的图像帧   tuple[bool, MatLike] 返回的是个元组（不可变list）
        ret, frame = video.read()
        if not ret:  # 如果 'ret' 为 False，表示没有读取到帧（可能是视频结束或发生错误）
            break  # 跳出循环，停止读取视频帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将视频帧转换为灰度图像
        cv2.imshow(name, gray)
        # 27 ESC 返回按下的键的 ASCII 码，与 是否等于q 的ASCII 码  ord("q") 。 waitKey(100) 等待键盘输入 不输入的话就继续运行 间接控制视频显示速度
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

# cv2_videoShow("video", "./videos/test.mp4")


def cv2_makeBorder(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 设置四个边框的大小，均为 50 像素
    top_size, bottmom_size, left_size, right_size = 50, 50, 50, 50

    # 使用不同的边框类型创建带有边框的图像
    # BORDER_REPLICATE: 复制最外边缘的像素
    replicated = cv2.copyMakeBorder(
        image, top_size, bottmom_size, left_size, right_size, cv2.BORDER_REPLICATE)

    # BORDER_REFLECT: 反射法，即对感兴趣的图像中的像素在两边进行复制
    reflect = cv2.copyMakeBorder(
        image, top_size, bottmom_size, left_size, right_size, cv2.BORDER_REFLECT)

    # BORDER_REFLECT_101: 反射法，和BORDER_REFLECT不同的是，在最边缘像素的反向，不同与对称法
    reflect101 = cv2.copyMakeBorder(
        image, top_size, bottmom_size, left_size, right_size, cv2.BORDER_REFLECT_101)

    # BORDER_WRAP: 外包法，顾名思义，可以看做是对图像像素进行外包处理
    wrap = cv2.copyMakeBorder(
        image, top_size, bottmom_size, left_size, right_size, cv2.BORDER_WRAP)

    # BORDER_CONSTANT: 常量填充，图像外部边缘使用常量填充，value表示填充值
    constant = cv2.copyMakeBorder(
        image, top_size, bottmom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)

    # 显示原始图像及其带有不同边框类型的版本

    # 创建一个宽度为 12 英寸，高度为 6 英寸的图形窗口
    plt.figure(figsize=(12, 6))

    # subplot(231) 表示将图像分为 2 行 3 列，图像在第一行第一列
    # 灰度图的时候 不需要把image bgr 转成  plt rgb 模式
    # 彩色图 plt.imshow(cv2.cvtColor(constant, cv2.COLOR_BGR2RGB))  # 转换颜色通道从 BGR 到 RGB
    plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title(
        'Original'), plt.axis('off')
    plt.subplot(232), plt.imshow(replicated, "gray"), plt.title("Replicated")
    plt.subplot(233), plt.imshow(reflect, "gray"), plt.title("Reflect")
    plt.subplot(234), plt.imshow(reflect101, "gray"), plt.title("Reflect_101")
    plt.subplot(235), plt.imshow(wrap, "gray"), plt.title("Wrap")
    plt.subplot(236), plt.imshow(constant, "gray"), plt.title("Constant")

    plt.tight_layout()  # 调整布局以改善子图之间的间距
    plt.show()  # 显示所有子图


# cv2_makeBorder("./images/moon.jpg")


# 数值计算
def cv2_imageAdd():
    image1 = cv2.imread("./images/moon.jpg")
    image2 = cv2.imread("./images/girl.jpg")

    # 如果2个维度(行列(宽高)相同)的数组相加数值超过了255就会对256取余 即%255 +1  的值
    image3 = image1+10
    # 截取图像的 0-5行 0-无穷列 0(blue) 通道
    print(image1[:5, :, 0])
    print(image3[:5, :, 0])

    # cv2.add 函数 超过255的值 不会截断   就是255
    # shape (宽高不同的不能相加 需要resize) resize 参数 是列 行
    # (450, 450, 3)
    # (1200, 1069, 3)
    # (1200, 1069, 3)
    resied = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    print(image1.shape)
    print(resied.shape)
    print(image2.shape)

    print(cv2.add(resied, image2)[:5, :, 0])

    cv2.imshow("add image", cv2.add(resied, image2))
    cv2.waitKey(0)  # 等待键盘事件 毫秒级  参数0表示无限期等待， 3000表示图片显示3s 直到按下任意键。
    cv2.destroyAllWindows()  # 这行代码使用OpenCV的cv2.destroyAllWindows函数关闭所有OpenCV窗口。

# cv2_imageAdd()


# 图像融合 (2个图像叠加在一起)
def cv2_imageFusion():
    image1 = cv2.imread("./images/moon.jpg")
    image2 = cv2.imread("./images/girl.jpg")

    # resize 的其他方法
    # (0,0)：这是目标尺寸，指定了输出图像的宽度和高度。在这种情况下，(0,0) 表示输出尺寸将由下面的 fx 和 fy 参数决定，而不是显式指定的大小。
    # fy=0.5：0.5 表示将图像的宽度缩小一半。 fy=2
    # fx=2：表示将图像的高度放大两倍。
    # interpolation=cv2.INTER_LINEAR：这是默认插值方法，用于在调整大小过程中计算像素值。cv2.INTER_LINEAR 表示使用双线性插值，这是一种常见的插值方法，适用于缩小图像时。
    cv2.resize(image2, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    # 之后的参数 0：这是一个标量值，将加到混合后的图像上。通常用于调整亮度。如果不需要调整亮度，可以将其设置为 0。
    image3 = cv2.addWeighted(image1, 0.2, image2, 0.8, 0)

    # 绘制到内存中的图形对象
    plt.imshow(image3), plt.axis('off')

    # show 才能在图形界面显示
    plt.show()


cv2_imageFusion()
