##准备数据

- **josn_format**	ground-truth的json目录

- **results**	经过ctpn预测后的实验结果目录

- **gt.zip**	转换格式后的ground-truth目录的压缩文件

  - E.g. "gt_img_1.txt":
      ```
      32,0,366,0,366,45,32,45,QAQ
      798,13,1160,13,1160,68,798,68,QAQ
      29,70,341,70,341,123,29,123,QAQ
      540,88,880,88,880,138,540,138,QAQ
      ```

- **submit.zip**	转换格式后的results目录的压缩文件

  - E.g. "res_img_1.txt":
      ```
      690, 169, 907, 169, 907, 207, 690, 207
      19, 126, 256, 126, 256, 169, 19, 169
      690, 623, 907, 623, 907, 656, 690, 656
      690, 368, 888, 368, 888, 407, 690, 407
      ```

## Getting Started

Install required module

  ```Shell
  pip install Polygon3
  ```
Then run
  ```Shell
  bash -x run.sh -g=gt.zip -s=submit.zip
  ```

Result

```
The sum of gt-boxes in all pictures: 19
The sum of detect-boxes all pictures: 17
Origin:
recall:  0.789 precision:  0.882 hmean:  0.833
SIoU-metric:
iouRecall: 0.548 iouPrecision: 0.613 iouHmean: 0.579
TIoU-metric:
tiouRecall: 0.471 tiouPrecision: 0.605 tiouHmean: 0.53
```





