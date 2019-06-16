import os
import shutil
from functools import cmp_to_key

# def cmp_center(x,y):
#     return x[2][1]-y[2][1]
# def cmp_x(x,y):
#     return x[2][0]-y[2][0]
# def cmp_y(x,y):
#     return x[2][1]-y[2][1]

def processFormat(source_dir, target_dir):
    source_list = [file for file in os.listdir(source_dir) if file.endswith('txt')]
    for file_s in source_list:
        file_name = file_s.split('.')[0][-1]
        file_rename = 'res_img_' + file_name + str('.txt')
        oral_dir = os.path.join(source_dir, file_s)
        new_dir = os.path.join(target_dir, file_rename)

        # content = []
        # with open(oral_dir,'r') as f_s:
        #     for line in f_s.readlines():
        #         x1,y1,x2,y2 = line.strip().split(',')
                # print(x1, y1, x2, y2)
                # content.append([[int(x1),int(y1)],[int(x2),int(y2)]])
            # print(content)
            # content.sort(key=cmp_to_key(cmp_center))
            # print(content)
        with open(oral_dir,'r') as f_s:
            with open(new_dir, 'w', encoding='utf-8') as f_t:
                for line in f_s.readlines():
                    x1, y1, x3, y3 = line.strip().split(',')
                    x2,y2 = x3,y1
                    x4,y4 = x1,y3
                    f_t.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(x1, y1, x2, y2, x3, y3, x4, y4))


if __name__ == '__main__':
    source_dir = './results/'
    target_dir = './submit/'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    processFormat(source_dir,target_dir)
    print("{} files Convert Finished!!!".format(len(os.listdir(target_dir))))