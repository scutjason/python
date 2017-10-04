# -*- coding: utf-8 -*-

"""
    用theano python库实现卷积
    参考博文 http://blog.csdn.net/niuwei22007/article/details/48025939
"""

# 先导入基本的库
import theano
import numpy
import pylab
from theano import tensor as T
from theano.tensor.nnet import conv

# 这里我们用了PIL的图像处理标准库,跟大名鼎鼎的opencv一样，不过pil是python库
from PIL import Image

# 生成一个随机数生成类rng，seed是23455
rng = numpy.random.RandomState(23455)

# 实例化一个4D的输入tensor，这是一个象征的输入，相当于形参，调用时传一个实参
input = T.tensor4(name='input')

# 权值数组的shape(数组大小)，用来确定需要生产的随机个数
# 该值得大小为 2 X 3的矩阵，其中矩阵的每个元素都是 9 X 9的矩阵
w_shp = (2, 3, 9, 9)

# 每个权值的边界，用来确定需要产生的每个随机数的范围
w_bound = numpy.sqrt(3 * 9 * 9)

