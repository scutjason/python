# 逻辑回归函数
'''
    数据初始化
    有400张照片，这些照片不是人的就是狗的。
    每张照片是28*28=784的维度。
    D[0]是训练集，是个400*784的矩阵，每一行都是一张照片。
    D[1]是每张照片对应的标签，用来记录这张照片是人还是狗。
    training_steps是迭代上限。
'''
'''
from theano import tensor as T 
import numpy as np
from theano import config
from theano import Param
from theano import function 

from theano import shared
from theano import scan

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

'''
x是输入的训练集，是个矩阵，把D[0]赋值给它。
y是标签，是个列向量，400个样本所以有400维。把D[1]赋给它。
w是权重列向量，维数为图像的尺寸784维。
b是偏倚项向量，初始值都是0，这里没写成向量是因为之后要广播形式。
'''
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

'''
'''
# train
for i in range(train_steps):
    pred,err=train(D[0], D[1])
'''
#print("Final model:") 
#print(w.get_value(), b.get_value()) 
#print("target values for D:", D[1] ) 
#print("prediction on D:", predict(D[0]))

D=[1,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,1,1,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,
0,0,1,1,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,
1,1,0,1,1,1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0,1,1,0,0,1,1,1,0,1,0,0,1,0,1,1,0,
0,0,1,0,1,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,0,0,1,0,
0,1,0,0,1,1,0,1,1,1,1,1,1,0,0,1,0,1,0,0,0,1,1,0,1,1,0,0,0,1,1,1,0,1,0,1,0,
0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,0,1,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,
0,1,0,0,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,1,1,0,0,1,0,1,1,1,0,1,1,0,1,0,1,
1,0,1,1,1,0,0,1,1,1,1,0,1,0,1,1,0,1,0,1,1,0,0,1,0,0,1,0,1,1,0,1,1,1,0,0,0,
0,0,0,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,0,0,0,
0,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0,1,1,0,0,0,1,1,0,1,0,0,1,1,0,1]



ok=0
err=0
#result=predict(D[0])
result= [True,False,False,False,True,False,False,False,False,True,False,True,
False,False,True,False,True,True,True,False,True,False,False,False,
True,False,False,True,False,False,False,False,False,False,True,False,
True,False,False,True,True,False,False,True,False,True,True,True,
True,True,True,False,True,True,False,False,True,False,False,False,
False,False,False,True,False,True,False,True,False,False,False,False,
True,True,True,True,False,True,True,True,True,True,True,False,
True,True,False,True,True,False,False,False,True,False,False,True,
True,False,False,True,True,True,False,True,False,False,True,False,
True,True,False,False,False,True,False,True,True,True,False,False,
True,True,False,True,False,False,False,False,True,True,True,True,
False,True,False,True,False,True,False,True,True,True,False,True,
False,False,True,False,False,True,False,False,True,True,False,True,
True,True,True,True,True,False,False,True,False,True,False,False,
False,True,True,False,True,True,False,False,False,True,True,True,
False,True,False,True,False,False,True,False,True,False,False,False,
True,True,False,False,True,False,True,False,False,False,True,False,
True,False,True,True,False,False,False,False,False,True,False,False,
True,False,False,False,True,False,False,True,False,False,False,False,
True,True,True,True,True,False,False,False,True,True,False,False,
False,False,False,True,True,False,False,True,False,True,True,True,
False,True,True,False,True,False,True,True,False,True,True,True,
False,False,True,True,True,True,False,True,False,True,True,False,
True,False,True,True,False,False,True,False,False,True,False,True,
True,False,True,True,True,False,False,False,False,False,False,True,
True,True,False,True,True,True,True,True,True,True,False,True,
False,True,False,True,False,True,True,True,False,True,False,True,
False,True,True,True,True,True,False,False,False,False,False,True,
True,True,False,True,True,True,False,True,True,True,True,True,
False,True,True,True,False,True,True,False,False,False,False,True,
True,True,False,False,False,False,False,False,True,True,False,False,
True,True,False,False,False,False,False,False,True,True,True,True,
False,True,True,False,False,False,True,True,False,True,False,False,
True,True,False,True]
R=[]
for i in D:
    if i == 1:
        if result[i] == True:
            ok+=1
        else:
            err+=1
    else:
        if result[i] == False:
            ok+=1
        else:
            err+=1
print(ok)
print('the predict is {}'.format(ok/400))