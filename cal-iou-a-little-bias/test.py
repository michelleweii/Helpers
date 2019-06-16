import json
import sys, os
import numpy as np
import cv2
import pymysql
import requests as rp
import shutil
#--------------
#author: haiweidai(dhw)
#time: 2018/12/17
#to calc the iou measure of an image
#--------------
def fileNeedDetect(fileName, dirPath):
    image_end_list = ['.BMP', '.bmp', '.JPG', '.jpg', '.png', '.PNG']
    end_r = [1 if fileName.endswith(end_str) else 0 for end_str in image_end_list]
    json_r = [1 if os.path.isfile(dirPath + '/' +fileName.replace(end_str, '.json')) else 0 for end_str in image_end_list]
    return sum(end_r) > 0 and sum(json_r) == len(image_end_list)

def imageEndsReplace(fileName, replaceStr):
    image_end_list = ['.BMP', '.bmp', '.JPG', '.jpg', '.png', '.PNG']
    for image_end in image_end_list:
        fileName = fileName.replace(image_end, replaceStr)
    return fileName

class IOUCalcMoudle:
#is if 2 points rect. ->generate to 4 point rect (left top, right top, right bottom, left bottom)
    def __init__(self, tDict, needSQL=False):
        # self.url_path_ = tDict['urlPath']
        # self.port_ = tDict['port']
        # self.db_name_ = tDict['dbName']
        # self.user_ = tDict['dbUser']
        # self.pwd_ = tDict['dbPwd']
        # self.test_p_batch_str_ = ""
        # #connect to db, create table
        # self.need_sql = needSQL
        # if self.need_sql:
        #     self.db_ = pymysql.connect(host=self.url_path_, \
        #     user=self.user_, password=self.pwd_, database=self.db_name_, port=int(self.port_))
        #     self.cusor_ = self.db_.cursor()
        #     sql = """CREATE TABLE IF NOT EXISTS DETECT_IOU (
        #     IDX INT(8) NOT NULL PRIMARY KEY AUTO_INCREMENT,
        #     USER_ID CHAR(100) NOT NULL,
        #     VERSION_ID CHAR(100) NOT NULL,
        #     FILE_NAME  CHAR(100) NOT NULL,
        #     IOU FLOAT,
        #     TS TIMESTAMP DEFAULT CURRENT_TIMESTAMP )"""
        #     self.cusor_.execute(sql)
        #     self.db_.commit()
        # self.post_url_ = tDict['postURL']
        # self.detect_type_ = tDict['detectType']
        # self.need_detected_img_ = tDict['needDetectedImg']
        self.batch_num_ = tDict['batchNum']
        # self.user_id_ =user_id_ tDict['userId']
        # self.version_id_ = tDict['versionId']

    # def __del__(self):
    #     if self.need_sql:
    #         self.db_.close()

    def gEdge(self, points):
        return [[points[0], points[1]], [points[1], points[2]], \
        [points[2], points[3]], [points[3], points[0]]]

    def regPoints(self, points, rowNum, colNum):
        return [[min(max(x[0], 0),  colNum - 1), min(max(x[1], 0), rowNum - 1)]  for x in points] 

#pre-deal the result fronm ground truth json file which may contained some result we need to regular
#input:
#gtPoints: the points inter from labelme json
#rowNum, colNum is the shape of the image
#output:
#the rect composed by edge
    def preDealGT(self, gtPoints, rowNum, colNum):
        len_p = len(gtPoints)
        #filter irregular point in gt points
        gtPoints = self.regPoints(gtPoints, rowNum, colNum)
        #generate 4 points by 2 points rect
        if len_p == 2:
            max_x = max(gtPoints[0][0], gtPoints[1][0])
            min_x = min(gtPoints[0][0], gtPoints[1][0])
            max_y = max(gtPoints[0][1], gtPoints[1][1])
            min_y = min(gtPoints[0][1], gtPoints[1][1])
            gtPoints = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
        return self.gEdge(gtPoints)

    def calcLineP(self, x, k, b):
        return round(k * x + b)
    
#calc the points in edge, result[1] == 0, k is normal; == 1, k is 0; == 2, k is inf
#input:
#edge is [[x1, y1], [x2, y2]]
#output:
#the point in this edge
    def gEdgePoints(self, edge):
        max_x = max(edge[0][0], edge[1][0])
        min_x = min(edge[0][0], edge[1][0])
        max_y = max(edge[0][1], edge[1][1])
        min_y = min(edge[0][1], edge[1][1])
        len_x = max_x - min_x
        len_y = max_y - min_y
        len_arr = max(len_x, len_y)
        if len_arr == 0:
            return [[], 3]
        inv_len = (1 / len_arr) * len_x
        #k is inf  
        if(edge[0][0] == edge[1][0]):
            return [[[edge[0][0], y] for  y in list(range(min_y, max_y + 1))], 2]
        #k is 0
        if(edge[0][1] == edge[1][1]):
            return [[[x, edge[0][1]] for x in list(range(min_x, max_x + 1))], 1]
        #calc k and b
        k = float(edge[0][1] - edge[1][1]) / float(edge[0][0] - edge[1][0])
        b = edge[1][1] - k * edge[1][0]
        #generate points by k and b
        return [[[round(idx * inv_len + min_x), self.calcLineP(idx * inv_len + min_x, k, b)] for idx in list(range(len_arr))], 0]

#generate points in one rect
#input:
#rect: one rect in image composed by edge
#rowNum, colNum as above
#output:
#the matrix of this matrix, the element in the matrix repsent is the pixel in rect
    def gRectMatrix(self, rect, rowNum, colNum):
        result_m = np.zeros((rowNum, colNum), dtype=int)
        edgeArr = np.ones((rowNum, 2), dtype=int) * -1
        for edge in rect:
            [edge_points, edge_type] = self.gEdgePoints(edge)
            if edge_type == 3:
                return result_m
            if edge_type != 1:
                x_l = [p[0] for p in edge_points]
                y_l = [p[1] for p in edge_points]
                edgeArr[y_l, 0] = [min(x1, x2) if x1!=-1 else x2 for (x1, x2) in zip(edgeArr[y_l, 0], x_l)]
                #add 1 : used in range generate, 
                edgeArr[y_l, 1] = [max(x1, x2) for (x1, x2) in zip(edgeArr[y_l, 1], x_l)]
            else:
                edgeArr[edge_points[0][1], 0] = edge_points[0][0]
                edgeArr[edge_points[0][1], 1] = edge_points[-1][0]
        edgeArr  = [data if data[1] == -1 else [data[0], data[1] + 1] for data in edgeArr]
        for (idx, edge_r) in enumerate(edgeArr):
            result_m[idx, edge_r[0]:edge_r[1]] = 1
        return result_m


#generate one class matrix for example for gt or predict rects
#input:
#rects: the rect list
#(rowNum, colNum) as above
#op the operate we want to generate the matrix of current class rects
#output:
#the matrix of current class rects
    def gRectsMatrix(self, rects, rowNum, colNum, op):
        if op == "sum":
            return sum([self.gRectMatrix(rect, rowNum, colNum) for rect in rects])
        else:
            result_m = np.zeros((rowNum, colNum), dtype=np.uint8)
            for rect in rects:
                result_m |= self.gRectMatrix(rect, rowNum, colNum)
            return result_m


#this function be used to calc the iou between ground truth rects and predict rects in one image
#input: gtDict , pDict is dict come from gt json and predict dict analysis by response json string, 
#(rowNum, colNum) is the shape of input image
#output:
#iou: the iou result
#gt_rects: the rects of ground truth
#p_rects: the rects if predict
#intersection_matrix: the matrix element is in intersection area between ground truth and predict result
    def calcIOU(self, gtDict, pTxtPath, rowNum, colNum):
        #rect is construct by list[leftTop, rightTop, rightBottom, leftBottom]
        gt_rects = list(map(lambda x: self.preDealGT(x['points'], rowNum, colNum), gtDict['shapes']))
        #the points in  predict maybe  float
        # pred
        p_rects = self.generateCoordinate(pTxtPath, rowNum, colNum)
        # p_rects = list(map(lambda x: self.gEdge(self.regPoints([\
        # [int(x['location']['leftTop']['x']), int(x['location']['leftTop']['y'])],\
        # [int(x['location']['rightTop']['x']), int(x['location']['rightTop']['y'])], \
        # [int(x['location']['rightBottom']['x']), int(x['location']['rightBottom']['y'])], \
        # [int(x['location']['leftBottom']['x']), int(x['location']['leftBottom']['y'])]],\
        #  rowNum, colNum)), pDict['areas']))
        #generate two matrix of gt and predict. which ind the points in their area
        print(gt_rects)
        print(p_rects)
        if len(gt_rects) == 0 or len(p_rects) == 0:
            return [0.0, gt_rects, p_rects, np.zeros((rowNum, colNum), dtype=np.uint8)]
        p_matrix = self.gRectsMatrix(p_rects, rowNum, colNum, "sum")
        gt_matrix = self.gRectsMatrix(gt_rects, rowNum, colNum, "sum")
        #this matrix used for draw result
        intersection_matrix = np.zeros((rowNum, colNum), dtype=np.uint8)
        gt_size = sum(sum(gt_matrix))
        p_size = sum(sum(p_matrix))
        intersection_size = sum(p_matrix[gt_matrix > 0])
        p_matrix[p_matrix > 0] = 1
        gt_matrix[gt_matrix > 0] = 1
        intersection_matrix = gt_matrix & p_matrix
        union_size = gt_size + p_size - intersection_size
        iou = float(intersection_size) / float(union_size)
        return [iou, gt_rects, p_rects, intersection_matrix]

#evaluate one case,  analysis gt json, predict json, labeled img save
#input:
#gtJsonFilePath: the path of ground truth labelme json file
#pDict: the dict analysis by predict json file
#imgFilePath: the path of img file
#dirPath: the dir of original data
#imgFileName: the img name of img
#outputPath: the dir we want to output our result
    def evaluateImgDetectResult(self, gtJsonFilePath, pTxtPath, imgFilePath, dirPath, imgFileName, outputPath):
        print(imgFileName)
        with open(gtJsonFilePath, 'r') as f_gt:
            gt_dict = json.load(f_gt)
            #total_p_dict = json.loads(pJsonStr)
            img = cv2.imread(imgFilePath, -1)
            (row_num, col_num, rgb) = img.shape
            p_dict = {}
            [iou, gt_rects, p_rects, intersection_matrix] = self.calcIOU(gt_dict, pTxtPath, row_num, col_num)
            if iou < 1.0:

                intersection_matrix = intersection_matrix.astype(np.uint8)
                #img[:,:,3] -= (intersection_matrix * 100)
                img[:,:,0] += (intersection_matrix * 10)
                list(map(lambda rect : list(map(lambda x: cv2.line(img, tuple(x[0]), tuple(x[1]), (255, 0, 0, 200), 1), \
                rect)), gt_rects))
                list(map(lambda rect : list(map(lambda x: cv2.line(img, tuple(x[0]), tuple(x[1]), (0, 0, 255, 200), 1), \
                rect)), p_rects))
                print(iou)
                t_arr = imgFileName.split('.')
                img_save_path = outputPath + '/' + imageEndsReplace(imgFileName, '_' + str(iou) + '_labeled.png')#t_arr[0] + '_' + str(iou) + '_labeled.png'
                cv2.imwrite(img_save_path, img)
            #for test-----------
            #bad_detect_path = '/Users/daihaiwei/Desktop/workSpace/data/truthData/testDir1' + '/' + t_arr[0] +\
            # '_' + str(iou) + '_labeled.png'
            #if iou < 0.5:
            #    cv2.imwrite(bad_detect_path, img)#    cv2.imwrite(bad_detect_path, img)#    cv2.imwrite(bad_detect_path, img)
            return iou

    def generateCoordinate(self,pTxtPath, rowsNum, colsNum):
        #data_list = [file for file in os.listdir(pTxtPath) if file.endswith('txt')]
        #for file in data_list:
            #file_dir = os.path.join(data_path, file)
        with open(pTxtPath, 'r') as f_read:
            content = []
            for line in f_read.readlines():
                x1, y1, x3, y3 = line.strip().split(',')
                x2, y2 = x3, y1
                x4, y4 = x1, y3
                points = self.regPoints([[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)], [int(x4), int(y4)]],\
                                        rowsNum, colsNum)

                content.append(self.gEdge(points))
            # print(content)
            return content

    # def getPredictDictByRequest(self, cases):
    #     f_stream_list = []
    #     for case in cases:
    #         f_stream_list.append(('files', open(case[1], 'rb')))
    #     file_data = f_stream_list#{'file': open(cases[0][1], 'rb')}
    #     post_data = {'detectType':self.detect_type_, 'needDetectedImg':self.need_detected_img_}
    #     p_d = rp.post(url=self.post_url_, files=file_data, data=post_data)
    #     p_json_str = p_d.text
    #     p_dict = json.loads(p_json_str)
    #     return p_dict

#batch evaluate cases, insert result into sql table detect_iou
#input : 
#cases: is a list , element is [[gtJsonFilePath, imgFilePath, dirPath, imgFileName, outputPath]]
#batchNum
    def BatchEvaluateDetectResult(self, cases, batchNum):
        #p_json_str = self.test_p_batch_str_#this is used to test, need to replace
        # if batchNum == 1 and len(cases) == 1:
        #     predict_json_path = imageEndsReplace(cases[0][1], '_predict.json')
        #     if os.path.isfile(predict_json_path):
        #         p_dict = json.load(open(predict_json_path, 'r'))
        #     else:
        #         p_dict = self.getPredictDictByRequest(cases)
        # else:
        #     p_dict = self.getPredictDictByRequest(cases)
        # t_case = []
        # for t_dict in p_dict['data']:
        #     for (idx, case) in enumerate(cases):
        #         if len(case) == 5 and case[3] == t_dict['fileName']:
        #             cases[idx].append(t_dict)
        #             t_case.append(cases[idx])
        #batch result str come from cases
        ious = [self.evaluateImgDetectResult(data[0], data[5], data[1], data[2], data[3], data[4])\
        for data in cases if len(data) == 6]
        return ious
        # if len(batch_insert) > 0:
        #     batch_insert_str = ",".join(batch_insert)
        #     sql = "replace into detect_iou(user_id, version_id, file_name, iou, ts) values" + batch_insert_str
        #     self.cusor_.execute(sql)
        #     self.db_.commit()

 #this is main function in this class
    def evaluate(self, dirPath, outputPath, predPath):
        json_files = os.listdir(dirPath)
        if len(json_files) == 0:
            print("there is no file in target dir")
        cases = []
        for file_name in json_files:
            #f_name_arr = file_name.split('.')
            gt_json_file = dirPath + '/' + imageEndsReplace(file_name, '.json')
            img_file = dirPath + '/' + file_name#f_name_arr[0] + '.png'
            pred_txt_path = os.path.join(predPath, 'res_' + imageEndsReplace(file_name, '.txt'))
            img_file_name = file_name #f_name_arr[0] + '.png'
            #img_save_file = dirPath
            if fileNeedDetect(file_name, dirPath): #file_name.endswith() len(f_name_arr) == 2 and f_name_arr[1] == 'json' and os.path.isfile(img_file):
                cases.append([gt_json_file, img_file, dirPath, img_file_name, outputPath, pred_txt_path])
        batch_len = int(len(cases) / self.batch_num_ + 1)
        if len(cases) % self.batch_num_ == 0:
            batch_len  -= 1
        batches = [ cases[idx * self.batch_num_ : min(((idx + 1) *  \
        self.batch_num_), len(cases))] for idx  in range(batch_len)]
        ious_arr = list(map(self.BatchEvaluateDetectResult, batches,  [self.batch_num_] * batch_len))
        mean_iou = np.mean([iou for ious in ious_arr for iou in ious])
        return mean_iou
    #this is function for test by  dhw
    def setTestPJson(self, batchJson):
        self.test_p_batch_str_ = batchJson

    #this is function for test by  dhw
    def testFun(self, dirPath, outputPath):
        files = os.listdir(dirPath)
        if len(files) == 0:
            print("there is no file in target dir")
        self.evaluate(dirPath, outputPath)

#this is function for test by dhw
# def test():
#     cfg_path = '/Users/daihaiwei/code_dir/model-text-detection-in-line/tools/calc-iou/calc_iou.cfg'
#     with open(cfg_path, 'r') as f_cfg:
#         cfg_dict = json.load(f_cfg)
#         iou_calc_m = IOUCalcMoudle(cfg_dict)
#         test_path = '/Users/daihaiwei/Desktop/workSpace/data/truthData/testCase/IOU_evaluation'
#         output_path = '/Users/daihaiwei/Desktop/workSpace/data/truthData/testCase/IOU_evaluation'
#         #p_json_list = ["", ""]
#         test_p_json_dir = '/Users/daihaiwei/Desktop/workSpace/data/truthData/testPJson'
#         p_json_str = open(test_p_json_dir + '/batch_test.json', 'r').read()
#         iou_calc_m.setTestPJson(p_json_str)
#         iou_calc_m.testFun(test_path, output_path)

if __name__ == "__main__":
    cfg_path = '/Users/admin/Documents/PyProject/cal-iou/cal-iou.cfg'
    with open(cfg_path, 'r') as f_cfg:
        cfg_dict = json.load(f_cfg)
        data_path = cfg_dict['dataPath']
        pred_path = cfg_dict["predPath"]
        output_path = cfg_dict['outputPath']
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        # 申明对象
        iou_calc_m = IOUCalcMoudle(cfg_dict)
        files = os.listdir(data_path)
        print(data_path, output_path)
        if len(files) == 0:
            print("there is no file in target dir")
        else:
            print('mean iou:', iou_calc_m.evaluate(data_path, output_path, pred_path))



        # iou_calc_m = IOUCalcMoudle(cfg_dict)
        # files = os.listdir(data_path)
        # if len(files) == 0:
        #     print("there is no file in target dir")
        # else:
        #     iou_calc_m.evaluate(data_path, output_path)
    #test()
        
        
        



    



    
