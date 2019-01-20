# 第一次产生的数据集，测试的class在最后一位。要将class放在第一位
import os

def change(source_dir, target_dir):
    source_list = os.listdir(source_dir)
    for file_s in source_list:
        print(file_s)
        with open(source_dir+file_s,'r') as f_s:
            with open(target_dir + file_s, 'w') as f_t:
            # f_s.write('vvv')
                for line in f_s.readlines():
                    # print(line)
                    x1, y1, x2, y2, class_name = line.strip().split(' ')
                        # f_t.write("bbbb")
                    f_t.write('{} {} {} {} {}\n'.format(class_name,x1, y1, x2, y2))


if __name__ == '__main__':
    source_dir = 'C:/Users/Michelle/Desktop/Clean_handwritten_sample_32/8000test/labels/'
    target_dir = 'C:/Users/Michelle/Desktop/Clean_handwritten_sample_32/8000test/ground-truth/'
    change(source_dir,target_dir)