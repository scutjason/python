'''
    date: 2017-09-21
    author: scutjason
'''

# softmax fisher
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load dataSets
mnist = input_data.read_data_sets("dataSets", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# sess
sess = tf.InteractiveSession()

# input x
x = tf.placeholder(tf.float32, [None, 784])
# label y_
y_ = tf.placeholder(tf.float32, [None, 10])

# w b
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# calc y
y = tf.nn.softmax(tf.matmul(x, w)+ b)

# cost  -sum(i y_i * log(yi)), every element *, so reduction_indices=[1]
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices =[1]))

# train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys})

# test  n*10
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))