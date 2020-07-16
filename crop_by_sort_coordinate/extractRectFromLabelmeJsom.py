import sys, os
import numpy as np
import shutil
import random as rd
import json
import cv2
#--------------
#author: michelleweii
#time: 2019/4/10
#extract rect from labelme data
#--------------
label_path = "20190409.txt"

def rmImg(img, rect):
    (row_num, col_num, rgb) = img.shape
    border_value = [np.mean(img[:, :, idx]) for idx in range(rgb)]
    img[rect[0][1]: rect[1][1], rect[0][0]: rect[1][0]] = border_value
    return img

def fileNeedDetect(fileName, dirPath, imgEndList):
    end_r = [1 if fileName.endswith(end_str) else 0 for end_str in imgEndList]
    json_r = [1 if os.path.isfile(dirPath + '/' +fileName.replace(end_str, '.json')) else 0 for end_str in imgEndList]
    return sum(end_r) > 0 and sum(json_r) == len(imgEndList)

def regPoints(points, rowNum, colNum):
    return [[min(max(x[0], 0),  colNum - 1), min(max(x[1], 0), rowNum - 1)]  for x in points]

#pre-deal the result fronm ground truth json file which may contained some result we need to regular
#input:
#gtPoints: the points inter from labelme json
#rowNum, colNum is the shape of the image
#output:
#the rect composed by [leftTop, rightBottom] 
def preDealGT(gtPoints, rowNum, colNum):
    len_p = len(gtPoints)
    #filter irregular point in gt points
    gtPoints = regPoints(gtPoints, rowNum, colNum)
    max_x = max(gtPoints[0][0], gtPoints[1][0])
    min_x = min(gtPoints[0][0], gtPoints[1][0])
    max_y = max(gtPoints[0][1], gtPoints[1][1])
    min_y = min(gtPoints[0][1], gtPoints[1][1])
    gtPoints = [[min_x, min_y], [max_x, max_y]]
    return gtPoints

def extractLabelmeRects(jsonPath, rowNum, colNum, labelList):
    with open(jsonPath, 'r') as f_gt:
        gt_dict = json.load(f_gt)
        gt_rects = [preDealGT(x['points'], rowNum, colNum) for x in gt_dict['shapes'] if x['label'] in labelList]
        return gt_rects

def cmp1(x,y):
    if x[0][0]==y[0][0]:
        return x[0][1]-y[0][1]
    else:
        return 1
def cmp2(x,y):
    return sum(x[0])-sum(y[0])
def cmp3(x,y):
    if sum(x[0])==sum(y[0]):
        return x[0][1]-y[0][1]
    else:
        return 1

from functools import cmp_to_key
def cmp_center(x,y):
    return x[2][1]-y[2][1]
def cmp_x(x,y):
    return x[2][0]-y[2][0]
def cmp_y(x,y):
    return x[2][1]-y[2][1]
# input: gt_rects [[[76, 563], [291, 602]], [[75, 4], [275, 47]],
# [[974, 780], [1158, 820]], [[973, 506], [1157, 551]],
# [[76, 508], [267, 554]], [[74, 1058], [267, 1104]], [[973, 566], [1188, 607]], [[974, 62],
# [1165, 108]], [[974, 625], [1164, 669]], [[973, 1065], [1158, 1102]],
# [[76, 345], [282, 391]], [[972, 842], [1172, 882]], [[77, 789], [260, 832]], [[76, 65], [260, 104]], [[76, 623], [282, 660]], [[73, 840], [264, 882]], [[974, 679], [1158, 720]], [[75, 291], [260, 325]], [[974, 119], [1164, 161]], [[973, 399], [1165, 436]], [[76, 958], [267, 1001]], [[75, 1009], [260, 1052]], [[76, 681], [268, 720]], [[525, 341], [709, 387]], [[75, 892], [259, 942]], [[973, 953], [1165, 992]], [[974, 171], [1166, 210]], [[75, 454], [260, 492]], [[76, 231], [260, 271]], [[525, 404], [716, 455]], [[76, 182], [260, 219]], [[524, 572], [716, 618]], [[973, 730], [1188, 773]], [[526, 781], [709, 824]], [[525, 1070], [709, 1111]], [[76, 402], [267, 443]], [[524, 6], [709, 44]], [[526, 288], [732, 327]], [[77, 125], [260, 164]], [[524, 844], [709, 887]], [[973, 452], [1157, 489]], [[973, 232], [1158, 276]], [[76, 730], [260, 774]], [[524, 64], [709, 102]], [[524, 901], [709, 945]], [[524, 1007], [723, 1053]], [[525, 181], [709, 221]], [[525, 710], [708, 768]], [[525, 450], [731, 494]], [[525, 515], [724, 561]], [[525, 959], [709, 996]], [[525, 680], [708, 739]], [[526, 612], [731, 657]], [[974, 290], [1188, 326]], [[526, 232], [740, 268]], [[973, 7], [1158, 49]], [[973, 1008], [1157, 1053]], [[973, 344], [1165, 384]], [[526, 121], [709, 155]], [[975, 895], [1238, 941]]]
def sort_rect(lista):
    maxheight = 0
    for item in lista:
        y = (item[1][1]+item[0][1])//2
        x = (item[1][0]+item[0][0])//2
        height = item[1][1]-item[0][1]
        maxheight = max(maxheight,height)
        item.append([x,y])
    lista.sort(key=cmp_to_key(cmp_center))
    row = []
    for i in range(len(lista)//3):
        row.append(lista[i * 3: 3 * (i + 1)])
    for i in row:
        i.sort(key=cmp_to_key(cmp_x))
    #for i in row:
    #    i.sort(key=cmp_to_key(cmp_y))
    new_rect = []
    for i in row:
        for j in i:
            new_rect.append(j[:2])
    return new_rect

#层遍得到所有满足fileNeedDetect的文件
def extractRectFromFile(dirPath, targetPath, txt_path):
    image_end_list = ['.BMP', '.bmp', '.JPG', '.jpg', '.png', '.PNG']

    if dirPath.endswith('DS_Store'):
        return 0
    trace_list = [[fileN, dirPath] for fileN in os.listdir(dirPath)]
    while trace_list:
        # print("tra:{}".format(trace_list))
        top = trace_list.pop()
        if top[0].endswith('DS_Store'):
            continue
        elif os.path.isdir(os.path.join(top[1], top[0])):
            curr_dir = os.path.join(top[1], top[0])
            t_list = [[fileN, curr_dir] for fileN in os.listdir(curr_dir)]
            t_list.extend(trace_list)
            trace_list = t_list
        elif fileNeedDetect(top[0], top[1], image_end_list):
            idx = 0
            img_name = os.path.join(top[1], top[0]) #1是路径，0是名字
            # print(img_name) # /Users/admin/Documents/crop/1/0_20190402154845021.jpg
            # print(top[0]) # 0_20190402154845021.jpg
            # print(top[1]) # /Users/admin/Documents/crop/1
            img = cv2.imread(img_name)
            [row, col, gbr] = img.shape
            stem, ext = os.path.splitext(top[0])

            target_img_path = os.path.join(targetPath, stem)
            if not os.path.exists(target_img_path):
                os.mkdir(target_img_path)
            txt_img_path = os.path.join(target_img_path, stem+'.txt')
            json_name = os.path.join(top[1], stem+'.json')
            gt_rects = extractLabelmeRects(json_name, row, col, ['r'])
            print('gt_rects',gt_rects)
            gt_rects = sort_rect(gt_rects)
            rects_coord = []
            for rect in gt_rects:
                rects_coord.append(rect)
                rect_row = rect[1][1] - rect[0][1]
                rect_col = rect[1][0] - rect[0][0]
                #if rect_row / rect_col < 0.75:
                #    continue
                # t_img = img[rect[0][1]: rect[1][1], rect[0][0]: rect[1][0]]
                t_img = img[rect[0][1]: rect[1][1], rect[0][0]: rect[1][0]]
                t_img_path = os.path.join(target_img_path, str(idx) + ext)
                print(t_img_path) # /Users/admin/Documents/crop/1/0_20190402154845021/0.jpg
                cv2.imwrite(t_img_path, t_img)
                idx += 1

            with open(txt_img_path,"w") as f2:
                with open(label_path,"r") as f1:
                    for i in sorted(rects_coord):
                        # print(i)
                        content = f1.readline()
                        f2.write(content)


        else:
            continue


if __name__=="__main__":
    # path = '/Users/daihaiwei/Desktop/yyt'#sys.argv[1] #"/Users/daihaiwei/Downloads/pic-all"
    # target_path = '/Users/daihaiwei/Desktop/n_data_1'#sys.argv[2] #"/Users/daihaiwei/Downloads/merged_pic"
    #

    filename = "1"
    path = '/Users/admin/Documents/PyProject/crop/20190409'#os.getcwd()+'/'+filename
    target_path = "/Users/admin/Documents/PyProject/crop/2019target"
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)

    os.mkdir(target_path)
    txt_path = ""
    # txt_path = target_path + '/' + str(name.split('.')[0]) + '.txt'

    # os.mkdir(target_path)
    extractRectFromFile(path, target_path, txt_path)

