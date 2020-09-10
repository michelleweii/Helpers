import os
import collections
import pandas as pd
import datetime
from cal_mAP import cal_mAP


def CalAccuracy(gt_dir, predict_dir):
    gt_list = os.listdir(gt_dir)
    predict_list = os.listdir(predict_dir)
    # print(gt_list)
    # print(predict_list)
    if 'desktop.ini' in gt_list:
        gt_list.remove('desktop.ini')
    if gt_list != predict_list:
        print("the number of file is not equal!!")
        return
    # print(gt_list)
    # print(predict_list)
    count = 0
    below_errorFile = 0
    over_errorFile = 0
    mistake_errorFile = 0
    for gt_file in gt_list:
        # print(gt_file)
        f_gt = open(gt_dir + gt_file, 'r')
        f_pred = open(predict_dir + gt_file, 'r')
        gt_cnt = len(f_gt.readlines())
        pred_cnt = len(f_pred.readlines())

        # print("gt_cnt:{}".format(gt_cnt))
        # print("pred_cnt:{}".format(pred_cnt))
        if gt_cnt != pred_cnt:
            # gt_cnt = len(f_gt.readlines())
            # pred_cnt = len(f_pred.readlines())
            if gt_cnt > pred_cnt:
                below_errorFile += 1
                print("below bounding boxes:{}".format(gt_file))
            # elif gt_cnt < pred_cnt:
            else:
                over_errorFile += 1
                print("over bounding boxes:{}".format(gt_file))
            f_gt.close()
            f_pred.close()
            continue
        else:
            gt, pred = [], []
            f_gt = open(gt_dir + gt_file, 'r')
            gt_cur = f_gt.readlines()
            f_pred = open(predict_dir + gt_file, 'r')
            pred_cur = f_pred.readlines()
            for gt_line in gt_cur:
                class_name1, x1, y1, x2, y2 = gt_line.strip().split(' ')
                gt.append(class_name1)
            for pred_line in pred_cur:
                class_name2, confidence, x1, y1, x2, y2 = pred_line.strip().split(' ')
                pred.append(class_name2)
            f_gt.close()
            f_pred.close()
            # print("gt:",gt)
            # print("pred:",pred)
            # print(collections.Counter(gt))
            # print(collections.Counter(pred))
            if collections.Counter(gt) == collections.Counter(pred):
                count += 1
            else:
                mistake_errorFile += 1
                print("predict error:{}".format(gt_file))

    print("1. predict the correct number of files: {}".format(count))
    print("2. more bbox error: {}".format(over_errorFile))
    print("3. less bbox error: {}".format(below_errorFile))
    print("4. misclassified error: {}".format(mistake_errorFile))
    accuracy = count / len(gt_list)
    print("5. accuracy: {}".format(accuracy))
    print("6. the total test number: {}".format(len(gt_list)))

    Result = {'total_number': len(gt_list),
              'correct_number': count,
              'more_bbox_error': over_errorFile,
              'less_bbox_error': below_errorFile,
              'misclassified_error': mistake_errorFile,
              'total_error': over_errorFile + below_errorFile + mistake_errorFile,
              'accuracy': accuracy,
              'test_time': end_date}

    return Result


if __name__ == '__main__':
    pwd = os.getcwd()
    print(pwd)
    gt_dir = pwd + '/ground-truth/'
    print(gt_dir)
    predict_dir = pwd + '/predicted/'

    end_date = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')  # 获取当前时间

    acc_result = CalAccuracy(gt_dir, predict_dir)
    print(end_date)
    ap_result = cal_mAP()
    result = dict(acc_result, **ap_result)

    columns = ['test_time', 'total_number', 'correct_number',
               'more_bbox_error', 'less_bbox_error',
               'misclassified_error', 'total_error', 'accuracy',
               '0','1','2','3','4','5','6','7','8','9','mAP']

    DataFrame = pd.DataFrame([result])

    ResultPath = 'accuracy_mAP.csv'
    if os.path.exists(ResultPath):
        DataFrame.to_csv(ResultPath, sep=',', columns=columns,
                         header=None, mode='a', index=0)
    else:
        DataFrame.to_csv(ResultPath, sep=',', columns=columns, index=0)
