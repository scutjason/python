# -*- coding:utf-8 -*-

"""
    theano 库学习
"""

# theano基本数据类型   所有的类型都在tensor 子模块中
"""
    tensor 矩阵分析里面的张量，你可以理解为就是高维数组，二阶张量就是矩阵，一阶张量就是向量。
    tensor 可以表示点、向量等 
    scalar （一个数据点，标量）， vector (向量）， matrix (矩阵）， tensor3 (三维矩阵)， tensor4 （四位矩阵）
    
    以b开头，表示byte类型（bscalar, bvector, bmatrix, brow, bcol, btensor3,btensor4）
    以w开头，表示16-bit integers（wchar）（wscalar, wvector, wmatrix, wrow, wcol, wtensor3, wtensor4）
    以i开头，表示32-bit integers（int）（iscalar, ivector, imatrix, irow, icol, itensor3, itensor4）
    以l开头，表示64-bit integers（long）（lscalar, lvector, lmatrix, lrow, lcol, ltensor3, ltensor4）
    以f开头，表示float类型（fscalar, fvector, fmatrix, fcol, frow, ftensor3, ftensor4）
    以d开头，表示double类型（dscalar, dvector, dmatrix, dcol, drow, dtensor3, dtensor4）
    以c开头，表示complex类型（cscalar, cvector, cmatrix, ccol, crow, ctensor3, ctensor4）

    int8	int16	int32	int64	float64     float32     complex64   complex128  complex128
    
    row (1xN matrix) 
    column (Mx1 matrix)
    broadcastable  一方面指定了类型的大小，另一方面也指定了那个维度上size只能为1
                  (True,表示对应维度上长度只能为1，matlab: size(A,broadcasrable(i)==True)=1)
                []                      scalar 
                [True]                  1D scalar (vector of length 1) 
                [True, True]            2D scalar (1x1 matrix) 
                [False]                 vector 
                [False, False]          matrix 
                [False] * n             nD tensor 
                [True, False]           row (1xN matrix) 
                [False, True]           column (Mx1 matrix) 
                [False, True, False]    A Mx1xP tensor (a) 
                [True, False, False]    A 1xNxP tensor (b) 
                [False, False, False]   A MxNxP tensor (pattern of a + b) 


"""
# In Theano, all symbols must be typed. 

from theano import tensor as T 
import numpy as np
from theano import config

                                            ##############################
                                            #                            #
                                            #           基础篇           #
                                            #                            #
                                            ##############################
# 申明一个标量        
# 三种方式
x=T.dscalar('x')                                                    # dscalar 表示类型
x=T.scalar('x', dtype='float64')                                    # scalar 表示标量，dtype表示类型
x=T.TensorType(dtype='float64', broadcastable=())('x')                # 也可以用TensorType来申明，自己定义类型
print(type(x))                                                      # <class 'theano.tensor.var.TensorVariable'>


# -------- function
from theano import Param
from theano import function 

x,y=T.dscalars('x','y')
z = x+y
f = function([x,y], z)          # [x,y] 是输入变量，z是输出变量
print(f(2, 3))

# eval()
# 也可以用变量的eval()方法，它接收一个dict作为一个字典
b=np.allclose(z.eval({x:16.3, y:12.1}), 28.4)
print(b)            # true

"""
    function 是theano中极其重要的一个函数，尤其是深度学习中，几乎都是function和scan
    function函数里面最典型的4个参数就是inputs,outputs,updates和givens
    
    inputs:     输入变量是list，里面是传给outputs的参数
    
    outputs:    输出参数列表，list或者dict。如果是dict，那么key必须是字符串。
    
    updates:    一组可迭代更新的量 (shared_variable, new_expression)的形式
                对其中的shared_variable输入用new_expression表达式更新，而这个形式可以是列表，元组或者有序字典
                updates其实也是每次调用function都会执行一次，则所有的shared_variable都会根据new_expression更新一次值。
                
    givens:     里面存放的是可迭代量，可以是列表，元组或者字典。每次调用function，givens的量都会迭代变化。
                它跟inputs一样也是作为参数传递给outputs的
                
"""
# 默认参数

x,y=T.dscalars('x','y')
z=x+y
f1=function([x, Param(y,default=1,name='by_name')],z)
print(f1(33))
print(f1(33,2))
print(f1(33,by_name=3))

# 共享变量
'''
    为了使GPU调用这些变量时，遇到一次就要调用一次，这样就会花费大量时间在数据存取上，导致使用GPU代码运行很慢，甚至比仅用CPU还慢。
    
    共享变量的类型必须为floatX
    
    shared 变量可以作为函数将的可以访问的数据，可以用get_value， set_value两个函数访问和获取值
    
    shared 变量既可以作为符号变量，也可以作为共享变量
    
    本质上是一块稳定存在的存储空间，类似于全局变量，但与全局变量不同的是，它可以存储在显存中
    
    Theano函数可以在每次执行时顺便修改shared variable的值
    
    updatas中的share variable会在函数返回后更新自己的值。
    
'''
from theano import shared

data_x=[[1,1,1],[1,1,1],[1,1,1]] 
# np.array 与 np.asarray 都可以将结构化数据转为ndarray，但是array会copy出一个副本，占用新的内存，但asarray不会。
shared_x=shared(np.asarray(data_x, dtype=config.floatX))


# example
state=shared(0)
inc=T.iscalar('inc')
accumulator=function([inc], state, updates=[(state,state+inc)])  # 这三个参数分别是inputs，outputs，updates
print(state.get_value())
accumulator(1)
print(state.get_value())        # state的值在调用函数之后才刷新,也就是updates在outputs执行之后再执行。
accumulator(300)  
print(state.get_value())
state.set_value(-1)  
print(accumulator(3))
print(state.get_value())

# 如果在某个函数中，共用了这个共享变量，但是又不想变动它的值，那么可以使用given参数foo替代这个变量state。而旧的state不发生变化
fn_of_state = state * 2 + inc
foo = T.scalar(dtype=state.dtype)  
skip_shared = function([inc, foo],fn_of_state,  
                           givens=[(state,foo)])    # 用foo代替state，state不变
print(skip_shared(1, 3))
print(state.get_value())

# scan函数    具有loop的效果，不断地扫描更新
'''
 theano.scan(fn, sequences=None, outputs_info=None, non_sequences=None, n_steps=None, truncate_gradient=-1, go_backwards=False, 
                   mode=None, name=None, profile=False, allow_gc=None, strict=False)
                   
fn ：函数类型，scan的一步执行。除了 outputs_info ，fn可以返回sequences变量的更新updates。是一个lambda或者def函数
    fn的输入变量顺序为 sequences中 的变量，outputs_info的变量，non_sequences中的变量。
    如果使用了taps，则按照taps给fn喂变量，taps的详细介绍会在后面的例子中给出。

sequences ：scan进行迭代的变量；scan会在T.arange()生成的list上遍历，例如下面的polynomial 例子。

            http://blog.csdn.net/u014519377/article/details/54359048

outputs_info ：初始化fn的输出变量，和输出的shape一致；如果初始化值设为None表示这个变量不需要初始值。迭代出处的结果
                每进行一步scan操作，outputs_info中的数值会被上一次迭代的输出值更新掉
                
non_sequences ：fn函数用到的其他变量，迭代过程中不可改变（unchange）。

n_steps ：fn的迭代次数。

返回值：形如(outputs, updates)格式的元组类型
        outputs是一个theano变量，或者多个theano变量构成的list.每一个theano变量包含了所有迭代步骤的输出结果。
        updates是形如（var, expression）的字典结构，指明了scan中用到的所有shared variables的更新规则 。
'''
# scan --- examples A的K次方, A**k
from theano import scan
k=T.iscalar('k')
A=T.vector('A')

outputs,updates=scan(lambda result,A:result*A, 
                                non_sequences=A, outputs_info=T.ones_like(A), n_steps=k)

result=outputs[-1]              # result = outputs [-1]可以告诉theano只需要取最后一次迭代结果，theano也会对此做相应的优化
fn_Ak=function([A,k], result,updates=updates)   # 求了result后在求updates
print(fn_Ak(range(10),2))



# ---example tanh(x(t).dot(W) + b)
X=T.matrix('X')
W=T.matrix('W')
b_sym=T.vector('b_sym')

results, updates=scan(lambda v: T.tanh(T.dot(v,W)+b_sym), sequences=X)
compute_elementwise = function(inputs=[X,W,b_sym], outputs=[results])
x=np.eye(2,dtype=config.floatX)
w=np.ones((2,2),dtype=config.floatX)
b=np.ones((2),dtype=config.floatX)
b[1]=2
print(compute_elementwise(x,w,b)[0])

print(np.tanh(x.dot(w)+b))
print('==============')




# 伪随机数
'''
伪随机数,
如果你用 random.seed(22),就能看到每次开始程序时的随机数都是一样的.
所以你就能生成一模一样的随机数数列

指定种子 seed
        每次调用 random() 会生成不同的值，在一个非常大的周期之后数字才会重复。
        这对于生成唯一值或变化的值很有用，不过有些情况下可能需要提供相同的数据集，
        从而以不同的方式处理。对此，一种技术是使用一个程序来生成随机值，并保存这些随机值，
        以便通过一个单独的步骤另行处理。不过，这对于量很大的数据来说可能并不实用，
        所以 random 包含了一个 seed() 函数，用来初始化伪随机数生成器，使它能生成一个期望的值集。
        
        种子（seed）值会控制生成伪随机数所用公式产生的第一个值，由于公式是确定性的，
        改变种子后也就设置了要生成的整个序列。seed() 的参数可以是任意可散列对象。
        默认为使用一个平台特定的随机源（如果有的话）。否则，如果没有这样一个随机源，则会使用当前时间。
        
保存状态 get_state
        random() 使用的伪随机算法的内部状态可以保存，并用于控制后续各轮生成的随机数。继续生成随机数之前恢复一个状态，
        这会减少由之前输入得到重复的值或值序列的可能性。getstate() 函数会返回一些数据，
        以后可以用 setstate() 利用这些数据重新初始化伪随机数生成器。
        
选择随机元素 choice
        一个枚举值序列中选择元素，即使这些值并不是数字。random 包括一个 choice() 函数，可以在一个序列中随机选择。
        下面这个例子模拟抛硬币 10000 次，来统计多少次面朝上，多少次面朝下。
        import random  
        import itertools  
          
        outcomes = { 'heads':0,  
                     'tails':0,  
                     }  
        sides = outcomes.keys()  
          
        for i in range(10000):  
            outcomes[ random.choice(sides) ] += 1  # 随机选择sides
          
        print 'Heads:', outcomes['heads']  
        print 'Tails:', outcomes['tails']

排列 shuffle

采样 sample
        sample() 函数可以生成无重复值的样本，且不会修改输入序列
'''
from theano.tensor.shared_randomstreams import RandomStreams    # 只能用于CPU
srng=RandomStreams(seed=234)    # 种子
rv_u=srng.uniform((2,2))        # 均匀分布 size为(2,2) 每个元素都是均匀分布，uniform(self, size=(), low=0.0, high=1.0, ndimndim=None):  
rv_n=srng.normal((2,2))         # 正太分布 size位(2,2) 每个元素都是正太分布，normal(self, size=(), avg=0.0, std=1.0, ndim=None):
f=function([], rv_u)            # 每次调用都会更新rv_u
f_val0 = f()   
f_val1 = f()    
print(f_val0)
print(f_val1)

g=function([], rv_n, no_default_updates=True) # 如果以后一直用这组随机数，就不再更新 .也就是每次调用g(),都会产生相同的结果
g_val0 = g()   
g_val1 = g()   
print(g_val0)
print(g_val1)

nearly_zeros=function([], rv_u+rv_u-2*rv_u) # 一个randow变量在每次执行函数时只提取一个数  , 所以接近与0
print(nearly_zeros())

# 分别设置，使用.rng.set_value()函数
rng_val =rv_u.rng.get_value(borrow=True) # Get the rng for rv_u。 borrow=true共用一个内存空间, borrow 是借的意思
rng_val.seed(89234) # seeds thegenerator    重新设置一个种子 
rv_u.rng.set_value(rng_val,borrow=True)  

srng.seed(902340)  # 当然你也可以选择全局设置，使用.seed()函数  

state_after_v0 = rv_u.rng.get_value().get_state()   # 保存调用前的state  这是保存rng的state，get_value其实是从rv_u中获取这个rng
print(nearly_zeros())
v1 = f()                                            #第一个调用，之后state会变化  
rng = rv_u.rng.get_value(borrow=True)   
rng.set_state(state_after_v0)                       # 为其state还原 
rv_u.rng.set_value(rng, borrow = True)  
v2 = f()    # 回到了v1之前的那个状态  # v2 != v1输出更新后state对应的随机数  
v3 = f()    # v3 == v1再次更新又还原成原来的state了  
  
print(v1)   
print(v2)  
print(v3)  


import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams        # 可以用于GPU

class Graph:
    def __init__(self, seed=123):
        self.rng=RandomStreams(seed)
        self.y=self.rng.uniform(size=(1,))

g1=Graph(123)
f1=theano.function([],g1.y)

g2 = Graph(seed=987)  
f2 = theano.function([], g2.y)

print('By default, the two functionsare out of sync.')
print('f1() returns ', f1()) 
print('f2() returns ', f2())

# 输出不同的随机值
def copy_random_state(g1, g2):
    if isinstance(g1.rng, MRG_RandomStreams):   # 判断变量类型 isinstance
        # 类型判断：其第一个参数为对象，第二个为类型名或类型名的一个列表。其返回值为布尔型。
        g2.rng.rstate = g1.rng.rstate
    for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):#打包 zip(A,B),A B对应的位置的元素组合成tuple 
        su2[0].set_value(su1[0].get_value())# 赋值  
'''
x = [1, 2, 3]
y = [4, 5, 6]
xy = zip(x, y)
print xy            # [(1, 4), (2, 5), (3, 6)]
'''
print('We now copy the state of thetheano random number generators.')    
copy_random_state(g1, g2)                   # 输出相同的随机值  
print('f1() returns ', f1())        
print('f2() returns ', f2())


# 导数 T.grad
'''
    第一个参数是要求导的表达式，第二个参数是自变量，或者自变量组成的list。
'''
x=T.dscalar('x')
y=x**2
gy=T.grad(y,x)              # T.grad的第1个参数必须是标量
print(gy)

f=function([x],gy)
print(f(4))                 # 8

# 雅克比矩阵
x=T.dvector('x')
y=x**2
J, updates=scan(lambda i,y,x:T.grad(y[i],x), sequences=T.arange(y.shape[0]), non_sequences=[y,x])  # 用y.shape[0] 来迭代
f=function([x], J, updates=updates)
print(f([4,4]))

# 海森矩阵
x=T.dvector('x')
y=x**2
cost=y.sum()
gy=T.grad(cost,x)
H,updates=scan(lambda i,gy,x:T.grad(gy(i),x),sequences=T.arange(gy.shape[0]), non_sequences=[gy,x])
f=function([x], H, updates=updates)
print(f([4,4]))
                                            ##############################
                                            #                            #
                                            #           实战篇           #
                                            #                            #
                                            ##############################


# 逻辑回归函数
'''
    数据初始化
    有400张照片，这些照片不是人的就是狗的。
    每张照片是28*28=784的维度。
    D[0]是训练集，是个400*784的矩阵，每一行都是一张照片。
    D[1]是每张照片对应的标签，用来记录这张照片是人还是狗。
    training_steps是迭代上限。
'''
rng= np.random

N=400
feats=28*28
D=(rng.randn(N,feats), rng.randint(size=N, low=0, high=2))
train_steps=10000

# 声明tensor变量 ，做为初始化
x=T.matrix('x')
y=T.vector('y')
w=shared(rng.randn(feats), name='w')
b=shared(0.,name='b')       # 这里0. 是为了让shared变量为floatX，否则出现type error 
print('Inital model:')
#print(w.get_value(), b.get_value())


'''
x是输入的训练集，是个矩阵，把D[0]赋值给它。
y是标签，是个列向量，400个样本所以有400维。把D[1]赋给它。
w是权重列向量，维数为图像的尺寸784维。
b是偏倚项向量，初始值都是0，这里没写成向量是因为之后要广播形式。
'''
# 创建theano Graph
p_1=1/(1+T.exp(-T.dot(x,w) -b))
prediction=p_1>0.5
xent=-y*T.log(p_1) -(1-y)*T.log(1-p_1)          # Cross-entropy loss function
cost=xent.mean()+0.01*(w**2).sum()              # The cost to minimize
gw,gb=T.grad(cost,[w,b])                        # Compute the gradient of the cost 

train=function(
    inputs=[x,y],
    outputs=[prediction,xent],
    updates=((w,w-0.1*gw), (b,b-0.1*gb)))
    
predict=function(inputs=[x], outputs=prediction)



# train
for i in range(train_steps):
    pred,err=train(D[0], D[1])
    
print("Final model:") 
print(w.get_value(), b.get_value()) 
print("target values for D:", D[1] ) 
print("prediction on D:", predict(D[0]))  



