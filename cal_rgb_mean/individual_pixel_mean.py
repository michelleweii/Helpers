import os
import cv2
from numpy import *

img_dir_head = 'C:/Users/Michelle/Desktop/Handwritten_sample_32/images'
# img_list=os.listdir(img_dir)
# img_size=224
sum_r = 0
sum_g = 0
sum_b = 0
count = 0

for i in range(2, 6):
    img_dir = img_dir_head + "/" + str(i)
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        print(img_name)
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img=cv2.resize(img,(img_size,img_size))
        sum_r = sum_r + img[:, :, 0].mean()
        sum_g = sum_g + img[:, :, 1].mean()
        sum_b = sum_b + img[:, :, 2].mean()
        count = count + 1

sum_r = sum_r / count
sum_g = sum_g / count
sum_b = sum_b / count
img_mean = [sum_r, sum_g, sum_b]
print(img_mean)


# resize224-图片2的-[241.52871065389715, 241.52871065389715, 241.52871065389715]
# 未resize-图片2的-[241.5397558673491, 241.5397558673491, 241.5397558673491]

# resize224-图片3的-[234.84837912587577, 234.84837912587577, 234.84837912587577]
# 未resize-图片3的-[234.8647019489786, 234.8647019489786, 234.8647019489786]

# resize224-图片4的-[228.18345043885236, 228.18345043885236, 228.18345043885236]
# 未resize-图片4的-[228.2049195331626, 228.2049195331626, 228.2049195331626]

# resize224-图片5的-[221.5032294780371, 221.5032294780371, 221.5032294780371]
# 未resize-图片5的-[221.53005404081543, 221.53005404081543, 221.53005404081543]

# 统计
# resize224-总的-[231.5159424,231.5159424,231.5159424]
# 未resize-总的-[231.5348578,231.5348578,231.5348578]

# 程序
# resize224-总的-
# 未resize-总的-[231.53485784757342, 231.53485784757342, 231.53485784757342]
