import cv2
import numpy as np
import os, sys
# from skimage import transform

# import Image
import glob
import json
import random as rd
import re

# 对内容进行缩小或者放大的时候，需要注意图像是黑底还是白底，对应的膨胀和腐蚀是相反的

# src_dir = os.getcwd()  + "/" + "a"
src_dir = os.getcwd()  + "/" + "characters"
# tar_dir = os.getcwd()  + "/" + 'data_generate' +'/'+ "characters"
# tar_dir = os.getcwd()  + '/'+ "b"
tar_dir = os.getcwd()  + '/'+ "characters_resize"

def charactersErode(src_dir,tar_dir):
    # 读取图片
    for dirlist in os.listdir(src_dir):
        print("dirlist:{}".format(dirlist))
        if dirlist == '.DS_Store':
            continue
        file_path = src_dir + '/' + dirlist
        save_path = tar_dir + '/' + dirlist

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for img in os.listdir(file_path):
            print("img:{}".format(img))
            if img == '.DS_Store':
                continue
            img_path = file_path + '/' + img
            # print(img_path)
            # print(os.path.isfile(img_path))
            if os.path.isfile(img_path):
                src = cv2.imread(img_path,0)
                # print(src)
                print(src.shape)
                height = int(src.shape[0]*rd.uniform(1.1,1.4))
                # 0是高，1是宽
                width = int(height/src.shape[0] * src.shape[1])
                src = cv2.resize(src,(width,height))
                # 设置卷积核
                print(src.shape)
                # src = transform.rescale(src, rd.randint(1.1,1.4)).shape  # 放大为原来图片大小的2倍
                kernel = np.ones((2, 2), np.uint8)
                # 图像膨胀处理
                erosion = cv2.dilate(src, kernel)

                cv2.imwrite(save_path+ '/' + img,erosion)



if __name__ == '__main__':
    charactersErode(src_dir,tar_dir)
