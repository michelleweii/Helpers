import os
import cv2
import numpy as np

labels_dir_head = 'C:/Users/Michelle/Desktop/Handwritten_sample_32/8000-test/labels'

class_num = [[[],[]] for i in range(10)]
# print(class_num)


# for i in range(2, 6):
#     labels_dir = labels_dir_head + "/" + str(i)
labels_list = os.listdir(labels_dir_head)
# print(labels_list)
for label_name in labels_list:
    # print(label_name)
    with open(labels_dir_head+"/"+label_name,'r') as f:
            for line in f.readlines():
                x1, y1, x2, y2, class_name = line.strip().split(' ')
                # print(class_name)
                width = int(x2)-int(x1)+1
                height = int(y2)-int(y1)+1
                class_num[int(class_name)][0].append(width)
                class_num[int(class_name)][1].append(height)

# print(class_num)
sum_width_mean = 0
sum_height_mean = 0

for i in range(10):
    # if i != 1:
    sum_width_mean += np.mean(class_num[i][0])
    sum_height_mean += np.mean(class_num[i][1])
    print("{}_mean_width:{}".format(i, np.mean(class_num[i][0])))
    print("{}_mean_height:{}".format(i, np.mean(class_num[i][1])))

print("")
print("sum_width_mean:{}".format(sum_width_mean/10))
print("sum_height_mean:{}".format(sum_height_mean/10))