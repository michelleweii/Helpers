import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

a = np.reshape(range(48),(4,6,2))
print("ori",a)
sess = tf.InteractiveSession()
res = tf.split(a, axis=1, num_or_size_splits=[4,2])
# print(res)
b1 = res[0] # (4, 4, 2)
b2 = res[1] # (4, 2, 2)
print("part1",res[0].eval())
print("part2",res[1].eval())
# 求b2第二个维度平均
c = tf.reduce_mean(b2, axis=1, keepdims=True)
print(c) # shape=(4, 1, 2)
print(c.eval())

d = tf.concat([b1,c],axis=1)
print(d.eval())
print(d.shape)

"""
ori [[[ 0  1]
  [ 2  3]
  [ 4  5]
  [ 6  7]
  [ 8  9]
  [10 11]]

 [[12 13]
  [14 15]
  [16 17]
  [18 19]
  [20 21]
  [22 23]]

 [[24 25]
  [26 27]
  [28 29]
  [30 31]
  [32 33]
  [34 35]]

 [[36 37]
  [38 39]
  [40 41]
  [42 43]
  [44 45]
  [46 47]]]

part1 [[[ 0  1]
  [ 2  3]
  [ 4  5]
  [ 6  7]]

 [[12 13]
  [14 15]
  [16 17]
  [18 19]]

 [[24 25]
  [26 27]
  [28 29]
  [30 31]]

 [[36 37]
  [38 39]
  [40 41]
  [42 43]]]
part2 [[[ 8  9]
  [10 11]]

 [[20 21]
  [22 23]]

 [[32 33]
  [34 35]]

 [[44 45]
  [46 47]]]
  
Tensor("Mean:0", shape=(4, 1, 2), dtype=int64)
[[[ 9 10]]

 [[21 22]]

 [[33 34]]

 [[45 46]]]
[[[ 0  1]
  [ 2  3]
  [ 4  5]
  [ 6  7]
  [ 9 10]]

 [[12 13]
  [14 15]
  [16 17]
  [18 19]
  [21 22]]

 [[24 25]
  [26 27]
  [28 29]
  [30 31]
  [33 34]]

 [[36 37]
  [38 39]
  [40 41]
  [42 43]
  [45 46]]]
(4, 5, 2)
"""