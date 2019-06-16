import random
import cv2
import os
import math
import numpy as np
import random as rd
from skimage.util import random_noise
from skimage import exposure
import shutil
import json
import base64

# 通过保留图片中心的内容，从中心向外扩充20px，然后裁剪图像
# 增加高斯噪音
def addNoise(imgPath,outputPath):
    img = cv2.imread(imgPath)
    reImgName = imgPath.split('/')[-1].replace('.jpg','_gauss.jpg')
    shutil.copyfile(imgPath.replace('jpg', 'json'),\
                    os.path.join(outputPath, imgPath.split('/')[-1].replace('.jpg', '_gauss.json')))
    cv2.imwrite(os.path.join(outputPath,reImgName),random_noise(img, mode='gaussian', clip=True) * 255)
# 椒盐噪音
def addSaltAndPepper(inputPath,outputPath):
    p = rd.uniform(0.02, 0.04)
    img = cv2.imread(inputPath)
    SP_NoiseNum=int(p*img.shape[0]*img.shape[1])
    for i in range(SP_NoiseNum):
        randX= rd.randint(0,img.shape[0]-1)
        randY= rd.randint(0,img.shape[1]-1)
        if rd.randint(0,1)==0:
            img[randX,randY]=0
        else:
            img[randX,randY]=255
    reImgName = inputPath.split('/')[-1].replace('.jpg', '_salt.jpg')
    shutil.copyfile(inputPath.replace('jpg', 'json'),\
                    os.path.join(outputPath, inputPath.split('/')[-1].replace('.jpg', '_salt.json')))
    cv2.imwrite(os.path.join(outputPath, reImgName), img)

# 调整亮度
def changeLight(inputPath,outputPath):
    img = cv2.imread(inputPath)
    flag = random.uniform(0.5, 5)  # flag>1为调暗,小于1为调亮
    reImgName = inputPath.split('/')[-1].replace('.jpg', '_light.jpg')
    shutil.copyfile(inputPath.replace('jpg','json'),\
                    os.path.join(outputPath,inputPath.split('/')[-1].replace('.jpg','_light.json')))

    cv2.imwrite(os.path.join(outputPath,reImgName),exposure.adjust_gamma(img, flag))

# def generateImg(inputPath,outputPath):
#     img = cv2.imread(inputPath)
#     print(img.shape) # 行，列，通道

def regPoints(points, rowNum, colNum):
    return [[min(max(x[0], 0),  colNum - 1), min(max(x[1], 0), rowNum - 1)]  for x in points]

def preDealGT(gtPoints, rowNum, colNum):
    # len_p = len(gtPoints)
    #filter irregular point in gt points
    gtPoints = regPoints(gtPoints, rowNum, colNum)
    #generate 4 points by 2 points rect
    max_x = max([point[0] for point in gtPoints])
    min_x = min([point[0] for point in gtPoints])
    max_y = max([point[1] for point in gtPoints])
    min_y = min([point[1] for point in gtPoints])
    gtPoints = [min_x, min_y, max_x, max_y]
    return gtPoints

def imgExtractRects(labelmeJson, img):
    latexes = [area['label'] for area in labelmeJson['shapes'] if area['label'] != 'p']
    # print(latexes)
    can_used = [area['points'] for area in labelmeJson['shapes'] if area['label'] != 'p']
    result_rects = list(map(lambda x: preDealGT(x, img.shape[0], img.shape[1]), can_used))
    return result_rects,latexes
#-------------------------------------------------------------------------------------------

def generateShapeDict(rect, latex):
    return {"shape_type":"rectangle", "line_color":None, "points":rect, "fill_color":None, "label":latex}

def generateLabelmeStr(imgPath, outputPath, rects, latexes=[]):
    if len(latexes) == 0:
        latexes = ['r1'] * len(rects)
    elif (len(latexes) != len(rects)):
        return ''
    template_dict_ = json.load(open(imgPath.replace('jpg','json'), 'rb'))
    template_dict_["shapes"] = [generateShapeDict(rect, latex) for (rect, latex) in zip(rects, latexes) if
                                len(rect) > 0]
    template_dict_['imageData'] = \
        str(base64.b64encode(open(outputPath, 'rb').read()))[2:-1]
    return json.dumps(template_dict_, ensure_ascii=True, indent=0, separators=(',', ':'))


def cropImg(imgPath, bboxes, latexes, outputPath):
    '''
    裁剪后的图片要包含所有的框
    输入:
        img:图像array
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
    输出:
        crop_img:裁剪后的图像array
        crop_bboxes:裁剪后的bounding box的坐标list
    '''
    # ---------------------- 裁剪图像 ----------------------
    img = cv2.imread(imgPath)
    w = img.shape[1]
    h = img.shape[0]
    x_min = w  # 裁剪后的包含所有目标框的最小的框
    x_max = 1
    y_min = h
    y_max = 1
    for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])

    d_to_left = x_min - 1  # 包含所有目标框的最小框到左边的距离
    d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
    d_to_top = y_min - 1  # 包含所有目标框的最小框到顶端的距离
    d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

    # 随机扩展这个最小框
    # crop_x_min = int(x_min - random.uniform(0, d_to_left))
    # crop_y_min = int(y_min - random.uniform(0, d_to_top))
    # crop_x_max = int(x_max + random.uniform(0, d_to_right))
    # crop_y_max = int(y_max + random.uniform(0, d_to_bottom))
    crop_x_min = int(x_min - 20)
    crop_y_min = int(y_min - 20)
    crop_x_max = int(x_max + 20)
    crop_y_max = int(y_max + 20)
    # print(crop_x_min)
    # 确保不要越界
    crop_x_min = max(1, crop_x_min)
    crop_y_min = max(1, crop_y_min)
    crop_x_max = min(w, crop_x_max)
    crop_y_max = min(h, crop_y_max)
    # print(crop_x_min)

    crop_img = img[crop_y_min - 1:crop_y_max, crop_x_min - 1:crop_x_max]
    reImgName = imgPath.split('/')[-1].replace('.jpg','_crop.jpg')
    cv2.imwrite(os.path.join(outputPath,reImgName),crop_img)
    # ---------------------- 裁剪boundingbox ----------------------
    # 裁剪后的boundingbox坐标计算
    crop_bboxes = list()
    for bbox in bboxes:
        crop_bboxes.append([[bbox[0] - crop_x_min + 1, bbox[1] - crop_y_min + 1],
                            [bbox[2] - crop_x_min + 1, bbox[3] - crop_y_min + 1]])
    content = generateLabelmeStr(imgPath, os.path.join(outputPath,reImgName), crop_bboxes, latexes)
    jsonName = imgPath.split('/')[-1].replace('.jpg','_crop.json')
    with open(os.path.join(outputPath,jsonName),'w') as f_j:
        f_j.write(content)

# 旋转
def rotateImg(imgPath, bboxes, latexes, outputPath, angle=5, scale=1.):
    '''
    输入:
        img:图像array,(h,w,c)
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        angle:旋转角度
        scale:默认1
    输出:
        rot_img:旋转后的图像array
        rot_bboxes:旋转后的boundingbox坐标list
    '''
    # ---------------------- 旋转图像 ----------------------
    img = cv2.imread(imgPath)
    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 仿射变换
    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), borderMode=cv2.BORDER_REPLICATE)
    reImgName = imgPath.split('/')[-1].replace('.jpg','_rotate.jpg')
    cv2.imwrite(os.path.join(outputPath,reImgName),rot_img)
    # ---------------------- 矫正bbox坐标 ----------------------
    # rot_mat是最终的旋转矩阵
    # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
    rot_bboxes = list()
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))
        # 改变array类型
        concat = concat.astype(np.int32)
        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = int(rx) + 1
        ry_min = int(ry) + 1
        rx_max = int(rx) + int(rw)
        ry_max = int(ry) + int(rh)
        # 加入list中
        rot_bboxes.append([[rx_min, ry_min], [rx_max, ry_max]])
    content = generateLabelmeStr(imgPath, os.path.join(outputPath,reImgName), rot_bboxes,latexes)
    jsonName = imgPath.split('/')[-1].replace('.jpg','_rotate.json')
    with open(os.path.join(outputPath,jsonName), 'w') as f_j:
        f_j.write(content)


def main():
    inputdir = '/Users/admin/Documents/crop-image/image'
    outputdir = '/Users/admin/Documents/crop-image/output'
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.mkdir(outputdir)
    for image in [file for file in os.listdir(inputdir) if file.endswith('jpg')]:
        with open(os.path.join(inputdir,image).replace('jpg','json'), 'r', encoding='utf-8') as f_gt:
            gt_dict = json.load(f_gt)
            img = cv2.imread(os.path.join(inputdir,image))
            gtPoints,latexes = imgExtractRects(gt_dict,img)
            cropImg(os.path.join(inputdir,image),gtPoints,latexes,outputdir)
            # angle = rd.randint(-5,5)
            # rotateImg(os.path.join(inputdir,image),gtPoints,latexes,outputdir,angle)
        
        # addNoise(os.path.join(inputdir,image),outputdir)
        # addSaltAndPepper(os.path.join(inputdir,image),outputdir)
        # changeLight(os.path.join(inputdir,image),outputdir)
        



if __name__ == '__main__':
    main()