import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import tensorflow.examples.tutorials.mnist.input_data as input_data
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import os
import random
import math
from PIL import Image, ImageEnhance

# 读取mnist数据集
# mnist = input_data.read_data_sets("./mnist/", one_hot=True)
# X_train, y_train = mnist.train.images, mnist.train.labels
# print("Original size: ", X_train.shape, y_train.shape)
# X_try = X_train[0]
# print(X_try.shape)

# 读取图像
X_train = []
y_train = []
path = '../handwritten_digits/'
folders = os.listdir(path)
for folder in folders:
    files = os.listdir(path + folder)
    for file in files:
        if file.endswith('.bmp'):
            image = Image.open(path + folder + '/' + file)
            # 转成灰度图像
            image = image.convert('L')
            # 图像模糊度调正（参数小模糊，大锐化），这里目的是加粗
            image = ImageEnhance.Sharpness(image).enhance(0.1)
            # 图像对比度，目的是加深
            image = ImageEnhance.Contrast(image).enhance(2.0)
            # 原图像实在太细，再重复上面操作二次
            image = ImageEnhance.Sharpness(image).enhance(0.1)
            image = ImageEnhance.Contrast(image).enhance(2.0)
            image = ImageEnhance.Sharpness(image).enhance(0.1)
            image = ImageEnhance.Contrast(image).enhance(2.0)
            # 转换回矩阵
            image = np.array(image)
            # 图像加入训练集
            X_train.append(image)
            # 因为目标文件夹的首字是0-9，这样刚好可以取到类别标注
            y_train.append(folder[0])
# 转成numpy格式
X_train = np.array(X_train, dtype='float32')
y_train = np.array(y_train, dtype='uint8')
# print(X_train.shape)
# print(type(X_train))
# print(y_train.shape)

old_size = 25 # 原来的图片数字上下没有边距
new_size = 32
# samples for 2 to n numbers, you also need to change the numbers of each group
n = 5

# 样本大小调正到新尺寸（但不是直接放缩）
X_train_resized = np.zeros((X_train.shape[0], new_size * new_size))

for i in range(X_train.shape[0]):
    # 取一个样本图像25*25
    # 图片的放缩需要在图片的格式下进行，矩阵下不行（maybe）
    img_temp = Image.fromarray(X_train[i].reshape(old_size, old_size)) # 转为图片
    # 放缩到22*22至26*26之间
    # 为了和mnist一样保持上下边距，数字大小有些随机。
    rand_size = old_size - random.randint(-1, 3)
    img_temp = img_temp.resize(size=(rand_size, rand_size))
    img_temp = np.array(img_temp).reshape(rand_size, rand_size)
    # 上下至少留2个像素的白边，具体数值随机
    top_blank = random.randint(2, new_size - 2 - rand_size) # 数字上面留2-random值的边距
    bottom_blank = new_size - rand_size - top_blank
    # 加上下白边
    img_temp = np.concatenate((np.zeros(shape=(top_blank, rand_size)), img_temp,
                               np.zeros(shape=(bottom_blank, rand_size))), axis=0) # e.g. 24*32
    # 左右基本平均留白边
    left_b = int((new_size - rand_size) / 2)
    right_b = new_size - rand_size - left_b
    # 加左右白边
    img_temp = np.concatenate((np.zeros(shape=(new_size, left_b)), img_temp,
                               np.zeros(shape=(new_size, right_b))), axis=1)
    # 处理好之后放入训练集中
    X_train_resized[i] = np.array(img_temp).reshape(new_size * new_size, )

print("Resized: ", X_train_resized.shape, y_train.shape)

# 这段费了好大劲，再添加的。
# 原因是原始样本图片存在不含或者几乎不含数字的图片（即空白图片）
# 因此，需要过滤掉。

flags = []
for i in range(X_train_resized.shape[0]): # 样本的个数
    flag = 0
    for j in range(X_train_resized.shape[1]): # 图片的像素点 == 32*32
        # 顺便把像素值从0-255转换到0-1，因为之前马代码的遗留，只能这么写了。
        X_train_resized[i][j] /= 255
        # 统计一下每张图像的像素值之和
        flag += X_train_resized[i][j]
    if flag < 10:
        # 记录像素值之和小于10的，认为是无用的样本
        flags.append(i)

print(len(flags))
# 全掉无用的样本，以及相应的标注
X_train_resized = np.delete(X_train_resized, flags, axis=0)
y_train = np.delete(y_train, flags, axis=0)

print("Bad samples deleted: ", X_train_resized.shape, y_train.shape)

# input("Stop")
# for i in range(X_new.shape[0]):
#     for j in range(X_new.shape[1]):
#         print(X_new[i][j])

# X_test, y_test = mnist.test.images, mnist.test.labels
# print(X_test.shape, y_test.shape)


# 和MNIST不同，这里不是one-hot编码，所以下面这段引掉
# def digit(y):
#     for i in range(10):
#         if (y_train[y][i] == 1):
#             return i

# 按0-9分开放置，每个类别样本数量不一样
X_train_samples = [[] for i in range(10)]
for i in range(X_train_resized.shape[0]):
    X_train_samples[y_train[i]].append(X_train_resized[i])

# print(type(X_train_samples))

# 看下每个类别样本的数量
for i in range(10):
    print(len(X_train_samples[i]))


# 以下两个def（第二个def包括4个子def）,目的是取框

# 把图片排成32行32列
def img_rebuild(eg):
    img_dot = [[] for i in range(new_size)]
    for i in range(new_size * new_size):
        img_dot[int(i / new_size)].append(eg[i])
    return img_dot

# 分别取数字在x和y轴上的最大和最小值
def img_xyminmax(eg):
    img_dot = img_rebuild(eg)

    def img_ymin():
        for i in range(new_size):
            for j in range(new_size):
                if (img_dot[i][j] > 0):
                    return i - 1 if i > 0 else 0

    def img_ymax():
        for i in range(new_size - 1, -1, -1):
            for j in range(new_size):
                if (img_dot[i][j] > 0):
                    return i + 1 if i < new_size - 1 else new_size - 1

    def img_xmin():
        for i in range(new_size):
            for j in range(new_size):
                if (img_dot[j][i] > 0):
                    return i - 1 if i > 0 else 0

    def img_xmax():
        for i in range(new_size - 1, -1, -1):
            for j in range(new_size):
                if (img_dot[j][i] > 0):
                    return i + 1 if i < new_size - 1 else new_size - 1

    return (img_xmin(), img_xmax(), img_ymin(), img_ymax())

# 合并数字，调用一次合并一个
def combine(ans, sample, position, str_number):
    this_xyminmax = img_xyminmax(sample) # 四个角的位置
    sample_rebuild = sample.reshape(new_size, new_size)
    # 以下先是在合并之前去掉白边
    # 去掉右白边
    move_1 = this_xyminmax[1]
    for i in range(new_size - move_1):
        sample_rebuild = np.delete(sample_rebuild, -1, axis=1)
    # 去掉左白边
    move_2 = this_xyminmax[0]
    for i in range(move_2):
        sample_rebuild = np.delete(sample_rebuild, 0, axis=1)

    # 以下决定合并的时候随机重叠度（可能重叠，也可能少许分离）
    # 在第一个小许白边之后，执行下面操作
    if (ans.shape[1] >= 20):
        # 如果不是第一个数字（1的间距要大一点）
        if position != 0:
            # 如果自己是1或者前面是1，那么随机重叠度设置小点
            if int(str_number[position]) == 1 or int(str_number[position - 1]) == 1:
                gap = random.randint(-2, 2)
            # 如果没遇到1，随机重叠度正常设置
            else:
                gap = random.randint(-5, 2)
        # 如果是第一个数字，无所谓是否是1，随机重叠度正常设置
        else:
            gap = random.randint(-5, 2)
    # 如果第一个小白边，不设置重叠度
    else:
        gap = 0  # 随机间隔
    # print(ans.shape,gap,this_xyminmax)

    # 根据上的取法可以得到调正原来的坐标数值
    x1 = ans.shape[1] + gap  # +(this_xyminmax[1]-this_xyminmax[0])/2
    y1 = this_xyminmax[2]
    x2 = x1 + this_xyminmax[1] - this_xyminmax[0]
    y2 = y1 + this_xyminmax[3] - this_xyminmax[2]

    # 加入位置list
    pos.append(x1)
    pos.append(y1)
    pos.append(x2)
    pos.append(y2)

    # 根据预选的随机重叠值，进行合并
    # 如果大于0，是分离操作
    if (gap >= 0):
        temp = np.zeros(shape=(new_size, gap))
        ans = np.concatenate((ans, temp, sample_rebuild), axis=1)
    # 如果小于0，是重叠操作
    else:
        temp1 = ans[:, gap:]
        temp2 = sample_rebuild[:, 0:-gap]
        if temp1.shape[1] != temp2.shape[1]:
            temp3 = np.zeros(shape=(new_size, temp1.shape[1] - temp2.shape[1]))
            temp2 = np.concatenate((temp2, temp3), axis=1)
        try:
            temp = temp1 + temp2
            for i in range(temp.shape[0]):
                for j in range(temp.shape[1]):
                    # 重叠操作，需要考虑加起来像素超过1的时候，置为1
                    if temp[i][j] > 1:
                        temp[i][j] = 1
        except:
            # 这个马留的，现在不会发生
            temp = temp1
            print("err")
        # 重叠操作（合并起来）
        ans = np.concatenate((ans[:, :gap], temp, sample_rebuild[:, -gap:]), axis=1)

    return ans

# 2-5位数的图片生成
for n_pic in range(2, n + 1):
    print("正在生成" + str(n_pic) + "个组合的数字...")
    for repeat in range(0, pow(10, n - n_pic)):
        # 2位数每种生成1000个，3位数每种生成100个
        # 4位数每种生成10个，5位数每种生成1个，各10万，合计40万
        print("第" + str(repeat) + "次循环开始...")
        num = -1  # 序号，这个变量其实可以不用的（马遗留）
        # 遍历每种数字
        for i in range(pow(10, n_pic)):
            # 加个2像素宽的小白边
            ans = np.zeros(shape=(new_size, 2))
            num += 1
            # 放置坐标值
            pos = []
            str_i = str(i).zfill(n_pic)
            for j in range(n_pic):
                # 当前数字的样本数量
                len_digit = len(X_train_samples[int(str_i[j])])
                # 随机选择一个进行合并
                ans = combine(ans,
                              X_train_samples[int(str_i[j])][random.randint(0, len_digit - 1)], j, str_i)  #
                # 遗留代码，无用
                # train1-3000，test3001-5000
                # [row, col] = ans.shape
                # print([row,col])
                # for row0 in range(row - 1):
                #     for col0 in range(col - 1):
                #         if (ans[row0, col0] >= 0.9):  # 二值化   #
                #             ans[row0, col0] = 0.9
                # if (ans[i, j] < 0.3):
                #   ans[i, j] = 0
            np.set_printoptions(threshold=np.inf)
            # left_blank = int(math.ceil((28 * n_pic - ans.shape[1]) / 2)) - 1
            # right_blank = int(((28 * n_pic - ans.shape[1]) / 2)) + 1
            # 随机白边设置
            left_blank = random.randint(0, new_size * n - ans.shape[1])
            if left_blank != 0:
                left_blank -= 1
            right_blank = new_size * n - ans.shape[1] - left_blank
            # 左右随机留白边操作，最终宽度为160
            ans = np.concatenate((np.zeros(shape=(new_size, left_blank)), ans,
                                  np.zeros(shape=(new_size, right_blank))), axis=1)

            # 保存标注
            save_label = "./DataSet/Handwritten_sample_" + str(new_size) + "/labels/" + str(n_pic) + "/"
            if not os.path.exists(save_label):
                os.makedirs(save_label)

            f = open(save_label + str(num).zfill(n_pic) + "_" + str(repeat) + ".txt", 'w')  # 写入txt文件
            for j in range(n_pic):
                f.write(str(pos[4 * j + 0] + left_blank) + " ")
                f.write(str(pos[4 * j + 1]) + " ")
                f.write(str(pos[4 * j + 2] + left_blank) + " ")
                f.write(str(pos[4 * j + 3]) + " ")
                f.write(str_i[j] + " ")
                f.write("\n")
            f.close()

            # 像素值从0-1转换回到0-255
            ans = np.trunc(255 - ans * 255)
            # 取整
            ans = np.array(ans).astype(np.uint8)
            # 转成图片
            img = Image.fromarray(ans)
            # 保存图片
            save_image = "./DataSet/Handwritten_sample_" + str(new_size) + "/images/" + str(n_pic) + "/"
            if not os.path.exists(save_image):
                os.makedirs(save_image)

            file_name = save_image + str(num).zfill(n_pic) + "_" + str(repeat) + ".png"
            img.save(file_name)

            # 弃用代码，本来用于生成高像素图像，但是用resize也可以啊！
            # plt.figure(figsize=(100, 100), dpi=1)
            # plt.imshow(ans, cmap=matplotlib.cm.binary, interpolation="nearest")  # 生成图片
            # plt.axis("off")  # 关闭坐标轴
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            # # plt.margins(0,0)
            # plt.savefig("./DataSet/Data4/images/" + str(num).zfill(5) + ".png",
            #             bbox_inches='tight', pad_inches=0)
            # # plt.show()
            # plt.close('all')
