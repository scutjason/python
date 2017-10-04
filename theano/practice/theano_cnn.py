# -*- coding: utf-8 -*-

"""
    theano 实现卷积神经网络
    
    参考博文 http://blog.csdn.net/u012162613/article/details/43225445
"""

# 导入必要的模块
import pickle as cPickle
import gzip
import os
import sys
import time
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool as downsample
from theano.tensor.nnet import conv

# 定义CNN的基本构件，卷积层、采样层、分类函数
"""
  卷积+下采样合成一个层LetNetConvPoolLayer
  rng：随机数生成器，用户初始化W
  input：4维向量，theano.tensor.dtensor4
  filter_shape：(number of filters, num input feature maps, filter height, filter width)
  image_shape：(batch size, num input feature maps, image height, image width)
  poolsize：(#row, #cols)
"""
class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
    
        # assert condition， condition为true，则继续往下执行，condition 为false时，停止执行
        # image_shape[1]和filter_shape(1)都是num input feature maps，它们必须是一样的
        assert image_shape[1] == filter_shape[1]
        self.input = input
        
        # 每个隐层神经元(即像素)与上层的连接数为num input feature maps * filter height
        fan_in = numpy.prod(filter_shape[1:])
        
        # lower layer 上每个神经元获得的梯度来自于 num input feature maps * filter height / pooling size
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize)
        
        # 以上求得fan_in、fan_out，将它们代入公式，依次来求随机初始值W, W就是线性卷积核
        W_bound = numpy.sqrt(6.0 / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        
        # The bias is 1D tensor -- one bias per feature map
        # 偏置b是一个一维向量，每个输出图的特征图对应一个偏置
        # 而输出图的个数有filter个数决定，因此用filter_shape[0]即num of filter来初始化
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        # 将输入的图像与filter做卷积， conv.conv2d函数
        # 卷积之后没有加b，然后通过sigmoid
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        
        # maxpooling， 最大采样子层
        pooled_out = downsample.pool_2d(
            input=conv_out,
            ws=poolsize,
            ignore_border=True
        )
        
        # 加偏置，在做tanh映射，得到卷积+采样子层的最终输出
        # 因为b是一维向量，这里用维度映射dimshuffer将其reshape。 比如b是(10,),
        # 则b.dimshuffle('x', 0, 'x', 'x') 将其reshape为(1, 10, 1, 1)
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        #卷积+采样层的参数
        self.params = [self.W, self.b]
    
    
"""
接下来定义隐藏层的类
隐藏层的输入是input，输出是神经元的个数。输入层与隐藏层是全连接的
假设输入是n_in维的向量（也可以说是n_in个神经元），隐藏层有n_out个神经元，因为是全连接。
所以一共有n_in*n_out个权重，故W大小是(n_in, n_out)，n_in行n_out列，每一列对应的隐藏层的每一个神经元的连接权重
b是偏置，隐藏层有n_out个神经元，故b是n_out维向量
rng是随机生成器，numpy.random.RandomState,用于初始化W
input训练模型所用到的所有输入，并不是MLP的输入层，MLP的输入层神经元个数是n_in，这里的参数input大小是(n_example, n_in), 每一行一个样本，即每一行一个MLP输入层
activition：激活函数，这里定义函数tanh  
"""
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activition=T.tanh):
        # 隐藏层的input即传进来的input
        self.input = input
        
        """
            代码要兼容GPU，则必须使用 dtype=theano.config.floatX, 并且定义为theano.shared
            另外，W的初始化有个规则：如果使用tanh函数，则在-sqrt(6. /(n_in + n_hidden))到sqrt(6./(n_in+n_hidden))之间均匀
            抽取数值来初始化W,若是sigmoid函数，则以上在乘4倍
        """
        # 如果W未初始化，则根据上述方法初始化
        # 因为有时可以用训练好的参数来初始化
        if W is None:
            W_values=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activition == theano.tensor.nnet.sigmoid:
                W_values *=4
            W = theano.shared(value=W_values, name='W', borrow=True)
            
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            
        #用上面定义的W、b来初始化类HiddenLayer的W、b
        self.W = W
        self.b = b
        
        # 隐含层的输出
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activition is None
            else activition(lin_output)
        )
        
        # 隐藏层的参数
        self.params = [self.W, self.b]
        
"""
定义分类器，Softmax回归

input，大小就是(n_example, n_in)， 其中n_example是一个batch的大小
因为我们训练用的Minibatch SGD，因此input 这样定义
n_in 即上一层的输出
n_out 输出的类别
"""

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
    
        # W的大小是n_in行n_out列，b为n_out维列向量。即输出对应W的列即b的一个元素
        self.W = theano.shared(
            value = numpy.zeros(
                (n_in, n_out),
                dtype = theano.config.floatX
            ),
            name = 'W',
            borrow = True
        )
        
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name = 'b',
            borrow = True
        )
        
        # input 是(n_example, n_in), W 是(n_in, n_out), 点乘得到(n_example, n_out), 加上偏置b,
        # 再作为T.nnet.softmax 的输入，得到p_y_given_x
        # 故p_y_given_x每一行代表每一个样本被估计为给类别的概率
        # b 是n_out维向量，与(n_example, n_out)矩阵相加，内部其实是先复制n_example个b，
        # 然后(n_example, n_out)矩阵的每一行都加上b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # argmax 返回最大值下标，因为本例数据集是MINIST，小标刚好就是类别，axis=1表示安行进行操作
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        # params， LogisticRegression的参数
        self.params = [self.W, self.b]
    
    # 代价函数NLL  
    # 因为我们是MSGD，每次训练一个batch，一个batch有n_example个样本，则y大小是(n_example,),  
    # y.shape[0]得出行数即样本数，将T.log(self.p_y_given_x)简记为LP，  
    # 则LP[T.arange(y.shape[0]),y]得到[LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,LP[n-1,y[n-1]]]  
    # 最后求均值mean，也就是说，minibatch的SGD，是计算出batch里所有样本的NLL的平均值，作为它的cost  
    def negative_log_likelihood(self, y):    
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        
    #batch的误差率  
    def errors(self, y):  
        # 首先检查y与y_pred的维度是否一样，即是否含有相等的样本数  
        if y.ndim != self.y_pred.ndim:  
            raise TypeError(  
                'y should have the same shape as self.y_pred',  
                ('y', y.type, 'y_pred', self.y_pred.type)  
            )  
        # 再检查是不是int类型，是的话计算T.neq(self.y_pred, y)的均值，作为误差率  
        #举个例子，假如self.y_pred=[3,2,3,2,3,2],而实际上y=[3,4,3,4,3,4]  
        #则T.neq(self.y_pred, y)=[0,1,0,1,0,1],1表示不等，0表示相等  
        #故T.mean(T.neq(self.y_pred, y))=T.mean([0,1,0,1,0,1])=0.5，即错误率50%  
        if y.dtype.startswith('int'):  
            return T.mean(T.neq(self.y_pred, y))  
        else:  
            raise NotImplementedError()
    
    # LeNet5=input+LeNetConvPoolLayer_1+LeNetConvPoolLayer_2+HiddenLayer+LogisticRegression+output

"""
    接下来是加载MINIS数据集
"""
def load_data(dataset):
    # dataset 是数据集的路径，程序首先检测该路径下有没有MINIST数据集，没有的话就去下载MINIS数据集
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # 检查dataset是不是目录
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(dataset) or data_file == 'mnist.pkl.gz':
            dataset = new_path
    
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.urlretrieve(origin, dataset)
        
    print('... loading data')
    
    # 从mnist.pkl.gz 中加载train_set， valid_set, test_set, 他们都是label
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f, encoding='bytes')
    f.close()
    
    # 将数据设置成shared variables， 主要是为了GPU加速，只有shared variable 才能存在GPU memory中
    # GPU 里数据类型只能是float   data_y是类别，所以最后转换为int返回
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(test_set_x, test_set_y), (valid_set_x, valid_set_y), (train_set_x, train_set_y)]
    return rval

        
"""
实现LeNet-5
LeNet-5有两个卷积层，第一个卷积层有20个卷积核，第二个卷积层有50个卷积核
"""
def evaluate_lenet5(learning_rate=0.1, n_epochs=200, dataset='mnist.pkl.gz', nkerns=[20,50], batch_size=500):
    """
     learning_rate:学习速率，随机梯度前的系数。 
     n_epochs 训练步数，每一步都会遍历所有batch，即所有样本 
     batch_size,这里设置为500，即每遍历完500个样本，才计算梯度并更新参数 
     nkerns=[20, 50],每一个LeNetConvPoolLayer卷积核的个数，第一个LeNetConvPoolLayer有 
     20个卷积核，第二个有50个 
    """
    rng = numpy.random.RandomState(23455)
    
    # 加载数据
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[0]
    test_set_x, test_set_y = datasets[0]
    
    # 计算batch的个数
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]  
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]  
    n_train_batches //= batch_size  
    n_valid_batches //= batch_size  
    n_test_batches //= batch_size 

    # 定义几个变量， index表示patch下标，x表示输入的训练数据，y是其对应标签
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    
    
    ########################
    #   BUILD ACTUAL MODEL #
    ########################
    print('... building the model')
    
    # 我们加载进来的batch大小的数据是(batch_size, 28 * 28)，但是LeNetConvPoolLayer的输入是四维的，所以要reshape
    layer0_input = x.reshape((batch_size, 1, 28, 28))
    
    # layer0即第一个LeNetConvPoolLayer层
    # 输入的单张图片(28,28)，经过conv得到(28-5+1 , 28-5+1) = (24, 24)
    # 经过maxpooling得到(24/2, 24/2) = (12, 12)
    # 因为每个batch有batch_size张图，第一个LeNetConvPoolLayer层有nkerns[0]个卷积核
    # 故layer0输出为(batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1 , 5, 5),
        poolsize=(2,2)
    )
    
    # layer1即第二个LeNetConvPoolLayer层  
    # 输入是layer0的输出，每张特征图为(12,12),经过conv得到(12-5+1, 12-5+1) = (8, 8),  
    # 经过maxpooling得到(8/2, 8/2) = (4, 4)  
    # 因为每个batch有batch_size张图（特征图），第二个LeNetConvPoolLayer层有nkerns[1]个卷积核  
    # 故layer1输出为(batch_size, nkerns[1], 4, 4)  
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        #输入nkerns[0]张特征图，即layer0输出nkerns[0]张特征图  
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2,2)
    )
    
    # 前面定义好了两个LeNetConvPoolLayer（layer0和layer1），layer1后面接layer2，这是一个全连接层，相当于MLP里面的隐含层  
    # 故可以用MLP中定义的HiddenLayer来初始化layer2，layer2的输入是二维的(batch_size, num_pixels) ，  
    # 故要将上层中同一张图经不同卷积核卷积出来的特征图合并为一维向量，  
    # 也就是将layer1的输出(batch_size, nkerns[1], 4, 4)flatten为(batch_size, nkerns[1]*4*4)=(500，800),作为layer2的输入。  
    # (500，800)表示有500个样本，每一行代表一个样本。layer2的输出大小是(batch_size,n_out)=(500,500)  
    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1]*4*4,
        n_out=500,
        activition=T.tanh
    )
    
    # 最后一层layer3是分类层，用的是逻辑回归中定义的LogisticRegression，  
    # layer3的输入是layer2的输出(500,500)，layer3的输出就是(batch_size,n_out)=(500,10)  
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    
    # 代价函数NLL
    cost = layer3.negative_log_likelihood(y)
    
    # test_model计算测试误差，x、y根据给定的index具体化，然后调用layer3，  
    # layer3又会逐层地调用layer2、layer1、layer0，故test_model其实就是整个CNN结构，  
    # test_model的输入是x、y，输出是layer3.errors(y)的输出，即误差。
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],  
            y: test_set_y[index * batch_size: (index + 1) * batch_size]  
        }  
    )  
    
    #validate_model，验证模型，分析同上。
    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x:valid_set_x[index * batch_size: (index+1)*batch_size],
            y:valid_set_y[index * batch_size: (index+1)*batch_size]
        }
    )
    
    # 下面是train_model，涉及到优化算法即SGD，需要计算梯度、更新参数  
    # 参数集
    params = layer3.params + layer2.params + layer1.params + layer0.params
    
    # 对各个参数的梯度
    grads = T.grad(cost, params)
    
    # 因为参数太多，在updates规则里面一个一个具体地写出来是很麻烦的，所以下面用了一个for..in..,自动生成规则对
    updates = [
        (param_i, param_i - learning_rate * grad_i)  
        for param_i, grad_i in zip(params, grads)  
    ]

    # train_model，代码分析同test_model。train_model里比test_model、validation_model多出updates规则
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x:train_set_x[index * batch_size: (index+1)*batch_size],
            y:train_set_y[index * batch_size: (index+1)*batch_size]
        }
    )
    ###############
    #   开始训练  #
    ###############
    print('....training')
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    
    validation_frequency = min(n_train_batches, patience/2)
    # 这样设置validation_frequency可以保证每一次epoch都会在验证集上测试
    
    #最好的验证集上的loss，最好即最小
    best_validation_loss = numpy.inf
    
    #最好的迭代次数，以batch为单位。比如best_iter=10000，说明在训练完第10000个batch时，达到best_validation_loss
    best_iter = 0
    
    test_score = 0
    start_time = time.clock()
    
    epoch = 0
    done_looping = False
        
    # 下面就是训练过程了，while循环控制的时步数epoch，一个epoch会遍历所有的batch，即所有的图片。  
    # for循环是遍历一个个batch，一次一个batch地训练。for循环体里会用train_model(minibatch_index)去训练模型，  
    # train_model里面的updatas会更新各个参数。  
    # for循环里面会累加训练过的batch数iter，当iter是validation_frequency倍数时则会在验证集上测试，  
    # 如果验证集的损失this_validation_loss小于之前最佳的损失best_validation_loss，  
    # 则更新best_validation_loss和best_iter，同时在testset上测试。  
    # 如果验证集的损失this_validation_loss小于best_validation_loss*improvement_threshold时则更新patience。  
    # 当达到最大步数n_epoch时，或者patience<iter时，结束训练 
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch +1
        for minibatch_index in range(n_train_batches):
            iter = (epoch -1) * n_train_batches + minibatch_index
            
            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            #cost_ij 没什么用，后面都没有用到,只是为了调用train_model，而train_model有返回值 
            
            if(iter +1) % validation_frequency  == 0:
                # compute zero-one loss on validation set 
                validation_losses = [validate_model(i) for i
                                        in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                
                print('epoch %i, minibatch %i/%i, validation error %f %%' %  
                    (epoch, minibatch_index + 1, n_train_batches, 
                    this_validation_loss * 100.))
                
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '  
                           'best model %f %%') %  
                          (epoch, minibatch_index + 1, n_train_batches,  
                           test_score * 100.))
                    
            if patience <= iter:
                done_looping = True
                break
                    
        end_time = time.clock()
        print('Optimization complete.')  
        print('Best validation score of %f %% obtained at iteration %i, '  
              'with test performance %f %%' %(best_validation_loss * 100., best_iter + 1, test_score * 100.))  
        print('The code ran for %.2fm' % ((end_time - start_time) / 60.))  


if __name__ == '__main__':
    evaluate_lenet5()