'''
    date:   2017-09-20
    author: scutjason
'''

'''
AlexNet for mnist model
    5   Convolutional Layer
    3   Full-connect  Layer
'''
import tensorflow as tf

'''
# get the dataSets
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("dataSets", one_hot = True)

# first, define the super_param
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

# define the network param
n_input = 784
n_classes = 10
dropout = 0.8

# define x, y
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# define the conv layer
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1,1,1,1], padding='SAME'), b), name=name)
    
# pool layer
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME', name=name)
    
# local response normalization
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)
    
def alex_net(_X, _weights, _biases, _dropout):
    
    # reshape to matrix, -1: python just calc itself
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
    # conv1
    conv1 = conv2d("conv1", _X, _weights["wc1"], _biases["bc1"])
    # pool1
    pool1 = max_pool("pool1", conv1, k=2)
    # norm1
    norm1 = norm("norm1", pool1, lsize=4)
    # dropout
    norm1 = tf.nn.dropout(norm1, _dropout)
    
    # conv2
    conv2 = conv2d("conv2", norm1, _weights["wc2"], _biases["bc2"])
    pool2 = max_pool("pool2", conv2, k=2)
    norm2 = norm("norm2", pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, _dropout)
    
    # conv3
    conv3 = conv2d("conv3", norm2, _weights["wc3"], _biases["bc3"])
    pool3 = max_pool("pool3", conv3, k=2)
    norm3 = norm("norm3", pool3, lsize=4)
    norm3 = tf.nn.dropout(norm3, _dropout)
    
    # fc1
    dense1 = tf.reshape(norm3, [-1, _weights["wd1"].get_shape().as_list()[0]])
    print(dense1.get_shape())
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights["wd1"]) + _biases["bd1"], name="fc1")
    
    # fc2
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights["wd2"]) + _biases["bd2"], name="fc2")
    
    out = tf.matmul(dense2, _weights["out"]) + _biases["out"]
    
    return out
    
weights = {
    "wc1": tf.Variable(tf.random_normal([3,3,1,64])),
    "wc2": tf.Variable(tf.random_normal([3,3,64,128])),
    "wc3": tf.Variable(tf.random_normal([3,3,128,256])),
    "wd1": tf.Variable(tf.random_normal([4*4*256, 1024])),
    "wd2": tf.Variable(tf.random_normal([1024,1024])),
    "out": tf.Variable(tf.random_normal([1024, 10]))
}  

biases = {
    "bc1": tf.Variable(tf.random_normal([64])),
    "bc2": tf.Variable(tf.random_normal([128])),
    "bc3": tf.Variable(tf.random_normal([256])),
    "bd1": tf.Variable(tf.random_normal([1024])),
    "bd2": tf.Variable(tf.random_normal([1024])),
    "out": tf.Variable(tf.random_normal([n_classes]))
}

# build the model, pred is the output class
pred = alex_net(x, weights, biases, keep_prob)

# loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))

# learn  let cost least
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# test 
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init variable
init = tf.global_variables_initializer()

# sess
with tf.Session() as sess:
    sess.run(init)
    step = 1
    
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y:batch_ys, keep_prob:dropout})
        
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y:batch_ys, keep_prob:1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.0})
            print("Iter" +str(step*batch_size) + ", Minibatch Loss = " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
        step +=1
    print("Optimization Finished!")
    
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.0}))

'''

# alexNet model
from datatime import datatime
import math
import time 

batch_size = 32
num_batches = 100

# print every layer
def print_activations(t):
    print(t.op.name, t.get_shape.as_list())
    
# model
def inference(images):
    parameters = []
    
    # conv1
    with tf.name_scope("conv1") as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3,96], dtpye = tf.float32, stddev=1e-1), name = 'weight')
        # output: 55*55*96, get rid of edge
        conv = tf.nn.conv2d(images, kernel, [1,4,4,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        # relu
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
        
    # lrn1
    # depth_radius = 4
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="lrn1")
    
    # pool1
    # feature_map: 27*27*96
    pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name="pool1")
    print_activations(pool1)
    
    # conv2
    with tf.name_scope("conv2") as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,96,256], dtype = tf.float32, stddev=1e-1), name = "weitht")
        # output: 27*27*256
        conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        # relu
        conv2 = tf.nn.relu(bias, name=scope)
        print_activations(conv2)
        parameters += [kernel, biases]
        
    # lrn2
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="lrn2")
    
    # pool2
    # feature_map: 13*13*256
    pool2 = tf.nn.max_pool(lrn2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID", name="pool2")
    print_activations(pool2)
    
    # conv3
    with tf.name_scope("conv3") as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,256,384], dtype=tf.float32, stddev=1e-1), name="weight")
        # output: 13*13*384
        conv = tf.nn.conv2d(pool2, kernel, [1,1,1,1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        # relu
        conv3 = tf.nn.relu(bias, name=scope)
        print_activations(conv3)
        parameters += [kernel, biases]
        
    # conv4
    with tf.name_scope("conv4") as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,384], dtype=tf.float32, stddev=1e-1), name="weight")
        # output: 13*13*384
        conv = tf.nn.conv2d(conv3, kernel, [1,1,1,1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        # relu
        conv4 = tf.nn.relu(bias, name = scope)
        print_activations(conv4)
        parameters += [kernel, biases]
        
    # conv5
    with tf.name_scope("conv5") as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256], dtype=tf.float32, stddev=1e-1), name="weight")
        # output: 13*13*256
        conv = tf.nn.conv2d(conv4, kernel, [1,1,1,1], padding="SAME")
        biases = tf.nn.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name="biases")
        bias = tf.bias_add(conv, biases)
        # relu
        conv5 = tf.nn.relu(bias, name=scope)
        print_activations(conv5)
        parameters += [kernel, biases]
        
    # pool5
    # feature_map: 6*6*256
    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID", name="pool5")
    print_activations(pool5)
    
    # fc1
    dense1 = tf.reshape(pool5, [-1, 6*6*256])
    wd1 = tf.nn.Variable(tf.truncated_normal([6*6*256, 4096], dtype=tf.float32, stddev=1e-1), name="wd1")
    bd1 = tf.nn.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name="bd1")
    dense1 = tf.nn.relu(tf.matmul(dense1, wd1) + bd1, name = "fc1")
    dense1 = tf.nn.dropout(dense1, keep_prob)
    
    # fc2
    wd2 = tf.nn.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1), name="wd2")
    bd2 = tf.nn.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name="bd1")
    dense2 = tf.nn.relu(tf.mutmal(dense1, wd2) + bd2, name="fc2")
    dense2 = tf.nn.dropout(dense2, keep_prob)
    
    # fc3
    wd3 = tf.nn.Variable(tf.truncated_normal([4096, 1024], dtype=tf.float32, stddev=1e-1), name="wd2")
    bd3 = tf.nn.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True, name="bd1")
    out = tf.matmul(dense2, wd3) + bd3
    
    return out

    



























