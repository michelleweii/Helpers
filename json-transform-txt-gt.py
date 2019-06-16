import json
import os
import cv2
import shutil

def regPoints(points, rowNum, colNum):
    return [[min(max(x[0], 0),  colNum - 1), min(max(x[1], 0), rowNum - 1)]  for x in points]

def preDealGT(gtPoints, rowNum, colNum):
    len_p = len(gtPoints)
    #filter irregular point in gt points
    gtPoints = regPoints(gtPoints, rowNum, colNum)
    #generate 4 points by 2 points rect
    max_x = max([point[0] for point in gtPoints])
    min_x = min([point[0] for point in gtPoints])
    max_y = max([point[1] for point in gtPoints])
    min_y = min([point[1] for point in gtPoints])
    gtPoints = [[min_x, min_y], [max_x, max_y]]
    # print(gtPoints)
    return gtPoints

def imgExtractRects(labelmeJson, img):
    can_used = [area['points'] for area in labelmeJson['shapes'] if area['label'] != 'x']
    result_rects = list(map(lambda x: preDealGT(x, img.shape[0], img.shape[1]), can_used))
    return result_rects

if __name__ == '__main__':
    json_dir = './json_format/'
    txt_dir = './gt/'
    if os.path.exists(txt_dir):
        shutil.rmtree(txt_dir)
    os.mkdir(txt_dir)
    json_file = [json_name for json_name in os.listdir(json_dir) if json_name.endswith('json')]
    # print(json_file)
    for file in json_file:
        idx = file.split('.')[0]
        # print(idx)
        img_name = idx + '.jpg'
        json_path = os.path.join(json_dir,file)
        txt_path = os.path.join(json_dir,img_name)
        img = cv2.imread(txt_path)
        with open(json_path, 'r', encoding='utf-8') as f_gt:
            gt_dict = json.load(f_gt)
            gtPoints = imgExtractRects(gt_dict,img)
        # print(gtPoints)
        with open(os.path.join(txt_dir,'gt_img_'+ idx+'.txt'),'w') as f_txt:
            for box in gtPoints:
                lefttop = box[0]
                rightbottom = box[1]
                f_txt.write('{},{},{},{},{}\n'.format(lefttop[0],lefttop[1],rightbottom[0],rightbottom[1],'###'))
                # print(lefttop[0],lefttop[1],rightbottom[0],rightbottom[1],'###')
    print("{} files Convert Finished!!!".format(len(os.listdir(txt_dir))))
