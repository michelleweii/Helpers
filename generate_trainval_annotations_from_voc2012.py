#!/usr/bin/evn python
# coding:utf-8
import os
import xml.etree.ElementTree as ET

# for train and val
AnnoPath = '../dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/'
JpegPath = '../dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'
train_AnnoPath_txt = '../sample/voc2012_annotations_train_set.txt'
val_AnnoPath_txt = '../sample/voc2012_annotations_val_set.txt'


train_txt = '../dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
val_txt = '../dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
# for test
# AnnoPath = '../dataset/test/VOCdevkit/VOC2012/Annotations/'
# JpegPath = '../dataset/test/VOCdevkit/VOC2012/JPEGImages/'
# AnnoPath_txt = '../sample/voc2012_annotations_test_set_HARD.txt'

HARD = False


##get object annotation bndbox loc start
def GetAnnotBoxLoc(AnotPath, t, d):  # AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet = {}  # 以目标类别为关键字，目标框为值组成的字典结构

    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(round(float(BndBox.find('xmin').text), 0)) - 1  # -1是因为程序是按0作为起始位置的
        y1 = int(round(float(BndBox.find('ymin').text), 0)) - 1
        x2 = int(round(float(BndBox.find('xmax').text), 0)) - 1
        y2 = int(round(float(BndBox.find('ymax').text), 0)) - 1

        if Object.find('truncated') is not None:
            Truncated = int(Object.find('truncated').text)
            if Truncated == 1:
                t += 1
        if Object.find('difficult') is not None:
            Difficult = int(Object.find('difficult').text)
            if Difficult == 1:
                d += 1
                if HARD is False:
                    continue

        if x1 >= x2 | y1 >= y2:
            print('Pleas stop, there is a mistake!')
            continue

        BndBoxLoc = [x1, y1, x2, y2]
        if ObjName in ObjBndBoxSet:
            ObjBndBoxSet[ObjName].append(BndBoxLoc)  # 如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
            ObjBndBoxSet[ObjName] = [BndBoxLoc]  # 如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧

    return ObjBndBoxSet, t, d


if __name__ == '__main__':

    train_list = []
    with open(train_txt, 'r') as f:
        for line in f:
            train_list.append(line.strip())
    print('There are {} training samples.'.format(len(train_list)))

    val_list = []
    with open(val_txt, 'r') as f:
        for line in f:
            val_list.append(line.strip())
    print('There are {} validation samples.'.format(len(val_list)))

    with open(train_AnnoPath_txt, 'w') as f:
        print('Train Converting Start...')

        t = 0
        d = 0
        row = 0

        Train_AnnoList = train_list
        Jpeg_files = os.listdir(JpegPath)

        for Anno in Train_AnnoList:
            # print(Anno)
            ObjBBoxes, t, d = GetAnnotBoxLoc(AnnoPath + Anno + '.xml', t, d)
            # print(ObjBBoxes)
            file_name = Anno + '.jpg'
            file_path_name = JpegPath + file_name
            if file_name not in Jpeg_files:
                print('The annotation and Jpeg do not match.')
                break

            for obj, bbox in ObjBBoxes.items():
                for i in range(len(bbox)):
                    row += 1
                    f.write('{},{},{},{},{},{}\n'.format(
                        file_path_name,
                        bbox[i][0],
                        bbox[i][1],
                        bbox[i][2],
                        bbox[i][3],
                        obj))
        print('Truncated: ', t)
        print('Difficult: ', d)
        print('There are {} objects.'.format(row))
        print('There are {} training samples.'.format(len(Train_AnnoList)))


    with open(val_AnnoPath_txt, 'w') as f:
        print('Val Converting Start...')

        t = 0
        d = 0
        row = 0

        Val_AnnoList = val_list
        Jpeg_files = os.listdir(JpegPath)

        for Anno in Val_AnnoList:
            # print(Anno)
            ObjBBoxes, t, d = GetAnnotBoxLoc(AnnoPath + Anno + '.xml', t, d)
            # print(ObjBBoxes)
            file_name = Anno + '.jpg'
            file_path_name = JpegPath + file_name
            if file_name not in Jpeg_files:
                print('The annotation and Jpeg do not match.')
                break

            for obj, bbox in ObjBBoxes.items():
                for i in range(len(bbox)):
                    row += 1
                    f.write('{},{},{},{},{},{}\n'.format(
                        file_path_name,
                        bbox[i][0],
                        bbox[i][1],
                        bbox[i][2],
                        bbox[i][3],
                        obj))
        print('Truncated: ', t)
        print('Difficult: ', d)
        print('There are {} objects.'.format(row))
        print('There are {} validation samples.'.format(len(Val_AnnoList)))

    print('The total sample are {}.'.format(len(Train_AnnoList)+len(Val_AnnoList)))

