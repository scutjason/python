# -*- coding:utf-8 -*-

'''
    theano实战
'''


################################################################################################################
#
#  单层的感知函数，是一个非0即1的两类的判断函数：计算出来的值大于一定的门限，就认为是等于1，否则就是0
#
################################################################################################################

# 写一个全连接层，实现前向传播、反向传播和参数更新。并用它实现一个3输入1输出的单层感知机，拟合函数y = x0 + x1 + x2。

import theano.tensor as T
import numpy as np
import theano
import pylab


class Dataset():
    def __init__(self, nsample, batchsize, nin):                # batchsize是指每次训练在训练集中取batchsize个样本训练；
        self.nsample=nsample                                    # 总的样本数 1024
        self.batchsize=batchsize                                # 每次从样本中，抽取的训练批次数目
        self.batch_num=nsample/batchsize
        self.data_x=np.array(np.random.rand(self.nsample*nin)).reshape(nsample,nin).astype('float32') # 1024*3
        print(self.data_x)
        print('===============================')
        self.data_y=self.data_x.sum(axis=1, keepdims=True)      # sum(axis=1)是将一个矩阵的每一行向量相加， y = x1 + x2 + x3 , w=[1,1,1], b=0
        print(self.data_y)
        
    def reset(self):
        self.batch_cnt=0
        self.indeces=np.array(range(self.nsample))  # 指数 等于index
        np.random.shuffle(self.indeces)     # 打乱
        self.data_x=self.data_x[self.indeces]
        self.data_y=self.data_y[self.indeces]
        
    def read(self):
        self.batch_cnt+=1
        if self.batch_cnt >= self.batch_num:
            return None
        batchsize = self.batchsize
        i=self.batch_cnt
        xsample=self.data_x[i*batchsize:(i+1)*batchsize]
        ysample=self.data_y[i*batchsize:(i+1)*batchsize]
        return xsample, ysample

Identity=lambda x:x
ReLu=lambda x:T.maximum(x, 0.0) # 返回 x与0.0 之间的最大值
Sigmoid=lambda x:T.nnet.sigmoid(x)
Tanh=lambda x:T.tanh(x)



class DenseLayer():
    '''
        定义了，参数 w和b，将这两个参数变成共享的，为了迭代
    '''
    def __init__(self, nin, nout, activation):
        self.activation=eval(activation)                                            # 字符串当成有效Python表达式来求值,并返回计算结果
        w_init=np.random.rand(nin*nout).reshape((nin, nout)).astype('float32')      # w 3*1
        b_init=np.random.rand(nout).reshape(nout).astype('float32')                 # b 1*1
        self.w=theano.shared(value=w_init, borrow=True)                             # 浅层复制，CPU共享
        self.b=theano.shared(value=b_init, borrow=True)
        
    def feedforward(self, x):
       return self.activation(T.dot(x, self.w) + self.b)                            # 调用函数 lambda x:x   入参是 w*x+b  返回这个计算结果
       

       
if __name__=='__main__':
    
    min=3
    nout=1
    
    model=DenseLayer(min,nout,activation='Identity')
    x=T.fmatrix('x')                    # 创建 输入x
    y=T.fmatrix('y')                    # 创建 标签y， y作为真实值
    out=model.feedforward(x)            # 返回w*x+b的预测值
    loss=T.sqr(y-out).mean()            # 均方差作为loss函数
    f_loss = theano.function([x, y], loss)  # 定义一个function来计算这个
    
    (grad_w,grad_b)=T.grad(loss,[model.w, model.b]) # 对这个loss求导  grad_w和grad_b是导数
    lr=T.fscalar('lr')
    f_update = theano.function([x,y,lr], [grad_w, grad_b],  # grad_w和grad_b    lr是学习率
                updates=[(model.w, model.w - lr * grad_w),  # 然后更新model.w ，更新的规则是 model.w - lr * grad_w
                         (model.b, model.b - lr * grad_b)], # 同时更新model.b ，更新的规则是 model.b - lr * grad_b
                allow_input_downcast=True)                                   # allow_input_downcast 允许输入显示转换
    
    epoch=2                         # 1个epoch等于使用训练集中的全部样本训练一次
    learn_rate=0.1
    batch_err=[]
    data=Dataset(nsample=1024, batchsize=64, nin=3)
    for epo in range(epoch):
        data.reset()
        while True:
            batchdata=data.read()
            if batchdata is None:
                break
            xsample, ysample = batchdata
            batch_error=f_loss(xsample, ysample)        # 计算batch损失
            f_update(xsample, ysample, learn_rate)      #　更新参数
            batch_err.append(batch_error)
            
    pylab.plot(batch_err, marker='o')
    pylab.show()
    
    
    
    
################################################################################################################
#
#  写一个卷积层ConvolutionLayer：  
#
################################################################################################################    
'''
    theano中有卷积函数。   output = TT.nnet.conv.conv2d(image, filter, image_shape, filter_shape)
    
    基本参数说明：
        image：          表示输入
        filter：         表示卷积核
        image_shape：    表示输入的shape
        filter_shape：   表示卷积核的shape
        border_mode:     valid：卷积之后图像尺寸会变小；full：卷积之后与原图保持一致（图像边缘会有填充padding）
        subsample：      下采样
        
    输入、卷积核和输出都是4维tensor。
    input:  (batchsize, nb_channel, nb_i_row, nb_i_column)      batchsize表示样本大小、nb_channel表示通道（RGB这些）、nb_i_row（图像的行）、nb_i_column（列）
    filter: (nb_filters,nb_channel, nb_f_row, nb_f_column)
    output: (batchsize, nb_filters, nb_o_row, nb_o_column)
    
'''

from theano.tensor.shared_randomstreams import RandomStreams

Identity=lambda x:x
ReLU=lambda x:T.maximum(x,0.0)
Sigmoid=lambda x:T.nnet.sigmoid(x)
Tanh=lambda x:T.tanh(x)

class ConvLayer():                 # 卷积层
    def __init__(self, image_shape, filter_shape, activation):
        self.activation=eval(activation)
        self.image_shape=image_shape
        self.filter_shape=filter_shape
        rng=RandomStreams(seed=42)
        w_init=np.asarray(rng.uniform(low=-1, high=1, size=filter_shape), dtype='float32')
        b_init=np.zeros(filter_shape[0], dtype='float32')       
        self.w=theano.shared(value=w_init, borrow=True)
        self.b=theano.shared(value=b_init, borrow=True)
        
        
    def feedforward(self, x):
        cout=theano.tensor.nnet.conv.conv2d(x, self.w, self.image_shape, self.filter_shape)
        output=self.activation(cout+self.b.dimshuffle('x', 0, 'x', 'x'))
        return output
    '''
    在feedforward里，conv2d返回值的shape是(batchsize, nb_filters, nb_o_row, nb_o_column)，
    然后我们要加上bias项b，但b的shape是(nb_filters)，不符合broadcast的规则，无法直接相加，所以，这就要用到dimshuffle了：
    
    dimshuffle: 变一个array张量结构的函数
    
    
    # 这样卷积结果和b都变成4维tensor，第1维上，二者size相同，其他维度上，b的size均为1，可以broadcast，这样就能愉快地相加了。
    # dimshuffle前
    b.shape == (nb_filters,)  # 其他维度为空

    # dimshuffle后,shape在'x'位置都被补1
    b = b.dimshuffle('x', 0, 'x', 'x')
    b.shape == (1, nb_filters, 1, 1)
    '''

# 这里介绍一下dimshuffle

'''
按照给定的pattern进行维度变换，通常patter里包含的是整数0, 1, ... ndim-1，
另外可以有'x'，这表示在当前位置增加一个可以broadcast的维度，即该维度size为1。
我感觉跟reshape类似

# 比如a的shape是(A,B,C,D)，则

a.dimshuffle(0,1,2,3)      # shape不变,依然是(A,B,C,D)
a.dimshuffle(0,1,3,2)      # shape变为(A,B,D,C)
a.dimshuffle(3,2,1,0)      # shape变为(D,C,B,A)
a.dimshuffle('x',3,2,1,0)  # 'x'位置加入size为1的维度,shape变为(1,D,C,B,A)
a.dimshuffle(3,2,'x',1,0)  # 'x'位置加入size为1的维度,shape变为(D,C,1,B,A)

'''



################################################################################################################
#
#  实战：RNN
#
################################################################################################################    

'''
    theano.ifelse
    
    output = theano.ifelse(condition, arg1, arg2)   # condition用int8表示，True则返回arg1，False返回arg2。
    
'''
from theano.ifelse import ifelse

x=T.fscalar('x')
y=T.fscalar('y')
z=ifelse(x>0, 2*y, 3*y)
f=theano.function([x,y], z)

print(f(1,10))


'''
    scan 函数
    
    result, updates = theano.scan(fn = 单步的函数,
                              sequences = 每步变化的值组成的序列,
                              outputs_info = 初始值
                              non_sequences = 每步不变的值)
                              
    outputs_info 会传入上一步的返回值
    result 每一项的符号表达式组成的list
'''


'''
    RNN  循环卷积网络
    
    老规矩，先讲讲历史和理论知识
    RNN，隐藏层之间也是相互有联系的，
    RNN在NLP自然语言处理上，取得了巨大成功。
    RNN最广泛最成功的模型便是LSTM（长短时记忆）模型
    
'''
from theano import scan

Identity=lambda x:x
ReLU=lambda x:T.maximum(x,0.0)
Sigmoid=lambda x:T.nnet.sigmoid(x)
Tanh=lambda x:T.tanh(x)

class SimpleRecurrentLayer():
    def __init__(self, rng, nin, nout, return_sequences):
        self.nin=nin
        self.nout=nout
        self.return_sequences=return_sequences
        w_init=np.asarray(rng.uniform(low=-1, high=1, size=(nin, nout)), dtype='float32')
        u_init=np.asarray(rng.uniform(low=-1, high=1, size=(nout, nout)), dtype='float32')
        b_init=np.asarray(rng.uniform(low=-1, high=1, size=(nout,)), dtype='float32')
        self.w=theano.shared(w_init, borrow=True)
        self.u=theano.shared(u_init, borrow=True)
        self.b=theano.shared(b_init, borrow=True)
    
    '''
        网络参数有三个：w、u、b。w是输入到h的输入矩阵，u是状态h之间的转移矩阵，b是偏置。
        return_sequences=True表示返回整个序列所有时刻对应的h(t)，return_sequences=False表示只返回最后时刻的h(t)
        
    '''
    def _step(self, x_t, h_t, w, u, b):
        h_next=x_t.dot(w)+h_t.dot(u)+b
        return h_next
        
    def feedforward(self, x):
        #assert len(x.shape) == 2   # x.shape: (time, nin)
        init_h=np.zeros(self.nout)
        results, updates=scan(fn=self._step, 
                                    sequences=x,
                                    outputs_info=init_h,
                                    non_sequences=[self.w, self.u, self.b])
        return results if self.return_sequences else results[-1]

class BatchRecurrentLayer():
    def __init__(self, rng, nin, nout, batchsize, return_sequences):
        self.nin=nin
        self.nout=nout
        self.batchsize=batchsize
        self.return_sequences=return_sequences
        w_init=np.asarray(rng.uniform(low=-1, high=1, size=(nin, nout)), dtype='float32')
        u_init=np.asarray(rng.uniform(low=-1, high=1, size=(nout, nout)), dtype='float32')
        b_init=np.asarray(rng.uniform(low=-1, high=1, size=(nout,)), dtype='float32')
        self.w=theano.shared(w_init, borrow=True)
        self.u=theano.shared(u_init, borrow=True)
        self.b=theano.shared(b_init, borrow=True)
        
    def _step(self, x_t, h_t, w, u, b):
        h_next=x_t.dot(w) + h_t.dot(u) +b
        return h_next
        
    def feedforward(self, x):
        #assert len(x.shape) == 3   # before shuffle, x.shape: (batchsize, time, nin)
        x=x.dimshuffle(1,0,2)
        init_h=np.zeros((self.batchsize, self.nout))
        results, updates = scan(fn=self._step, 
                                sequences=x,
                                outputs_info=init_h,
                                non_sequences=[self.w, self.u, self.b])
                
        if self.return_sequences:
            return results.dimshuffle(1, 0, 2)
        else:
            return results[-1]



if __name__ == '__main__':
    rng = np.random.RandomState(seed=42)
        
    nin=3
    nout=4
    
    # test.simpleRNN
    x=T.fmatrix('x')
    srnn=SimpleRecurrentLayer(rng, nin, nout, True)
    sout=srnn.feedforward(x)
    fs=theano.function([x], sout)
    
    xv=np.ones((8,3), dtype='float32')
    print(fs(xv).shape)

    
    # test batchRNN
    batchsize=4
    y=T.tensor3('y')
    brnn=BatchRecurrentLayer(rng, nin, nout, batchsize, True)
    bout=brnn.feedforward(y)
    fb=theano.function([y], bout)
    
    yv=np.ones((batchsize, 8,3), dtype='float32')
    print(fb(yv).shape)


#####################################################################################
#
#
#               GPU加速技巧
#
#####################################################################################

'''
并不是所有情况下GPU都会比CPU快，它只在并行计算上有优势，如果我们的代码在GPU上很慢，那就要找找问题了。

首先在CPU和GPU下运行示例程序：testing-theano-with-gpu，如果用到了GPU而且比CPU版本快，那就是自己代码的问题。可以参考这些tips：
    1、Theano只对float32下的运算加速。
    
    2、大矩阵的相乘、卷积、点运算等操作，如果能大到让30个处理器同时工作，那么GPU可以有5到50倍的加速。
    
    3、indexing、shuffling、reshape等操作GPU没有加速。
    
    4、对张量进行sum()等操作GPU比CPU稍慢，这个坑很多，比如训练语言模型计算最后的softmax，词表很大的话，速度一下子就慢了。
    
    5、大量的数据拷贝会降低速度。这条可能不太容易引起注意，但影响却很大，我曾经在一个程序里进行了几次大矩阵的stack和concatenate，
        跑起来比CPU还慢，改了实现方法以后GPU速度瞬间快了十几倍。
    
    6、最后，如果是频繁更新某个常驻变量，用shared variable存储，然后在function里用updates更新，比如神经网络的权值。

'''


