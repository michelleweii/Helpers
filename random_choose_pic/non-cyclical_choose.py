import os

import random

import shutil

import skimage.io as io


def copyFile(fileDir, tarDir):
    pathDir = os.listdir(fileDir)

    # for filename in pathDir:
    #     print(filename)

    coll = io.ImageCollection(str)

    # print(len(coll))  # 打印图片数量

    num = 500
    # num = int ((2*len(coll))/10)
    # print(num)

    sample = random.sample(pathDir, num)
    # print(sample)
    for name in sample:
        # print(name)
        # print(fileDir + name)
        label = name.replace('.png','.txt')
        print(name)
        print(label)
        shutil.copyfile(fileDir + name, tarDir + name)
        shutil.copyfile(docDir + label, tarDocDir + label)
        os.remove(fileDir + name)
        os.remove(docDir + label)



if __name__ == '__main__':
    fileDir = "C:/Users/Michelle/Desktop/HDSR_Dataset/S_HDS5/images/"  # 填写要读取图片文件夹的路径
    #add
    docDir = "C:/Users/Michelle/Desktop/HDSR_Dataset/S_HDS5/labels/"

    tarDir = "C:/Users/Michelle/Desktop/HDSR_Dataset/R_HDS5/images/"  # 填写保存随机读取图片文件夹的路径
    #add
    tarDocDir = "C:/Users/Michelle/Desktop/HDSR_Dataset/R_HDS5/labels/"


    str = 'fileDir*.png'  # fileDir的路径+*.jpg表示文件下的所有jpg图片

    copyFile(fileDir, tarDir)
