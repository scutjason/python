'''
    date:   2017-09-22
    author: scutjason
'''
# leNet-5  conv: 2  pool:2 fc:2
import tensorflow as tf
import numpy as np
import time
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

# define param
batch_size = 50
# it's too important, and no need large
learning_rate = 1e-4
trainning_step = 20000
dropout = 0.8
n_input = 784
n_classes = 10

# x
x = tf.placeholder(tf.float32, [None, n_input])
# y
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)



# LeNet-5 network
def leNet_5(input_tensor, _dropout):
    input = tf.reshape(input_tensor, shape=[-1, 28, 28, 1])
    
    # conv1  
    # input         28*28*1
    # kernel        5*5*32
    # feature_map   28*28*32
    with tf.variable_scope("conv1") as scope:
        kernel = tf.get_variable("weights", shape=[5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[32], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input, kernel, [1,1,1,1], padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias)
        
    # pool1
    # feature map   14*14*6 
    with tf.variable_scope("pool1") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name = "pool1")

    # conv2
    # kernel        5*5*64
    # feature_map   14*14*64
    with tf.variable_scope("conv2") as scope:
        kernel = tf.get_variable("weights", shape=[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases",  shape=[64], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, kernel, [1,1,1,1], padding="SAME")
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias)
        
    # pool2
    # feature_map   7*7*64
    with tf.variable_scope("pool2") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="pool2")
        
    # fc1
    # 1024
    with tf.variable_scope("dense1") as scope:
        pool = tf.reshape(pool2, [-1, 7*7*64])
        weights = tf.get_variable("weights", shape=[7*7*64,1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[1024], initializer=tf.constant_initializer(0.1))
        dense1 = tf.nn.relu(tf.matmul(pool, weights) + biases)
        # dropout layer
        dense1 = tf.nn.dropout(dense1, _dropout) 
    
    # out
    with tf.variable_scope("out") as scope:
        weights = tf.get_variable("weights", shape=[1024,10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[10], initializer=tf.constant_initializer(0.1))
        out = tf.matmul(dense1, weights) + biases

    return out
 
# pred 
pred = leNet_5(x, keep_prob)

# loss
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
y_conv = tf.nn.softmax(pred)
cross_entropy = -tf.reduce_mean(y * tf.log(y_conv))

# training
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# test
correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init
init = tf.global_variables_initializer()

# load dataSets
mnist = input_data.read_data_sets("dataSets", one_hot = True)

# save the w and b
saver = tf.train.Saver()

def run(argv):
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        if argv[1] == '1':
            while step < trainning_step:
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys, keep_prob:dropout})
                
                if step % 100 == 0:
                    acc = sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})
                    print("step %d, trainning accuracy %g" %(step, acc))
                step += 1
                
            # tensorbord
            writer = tf.summary.FileWriter('./scope_leNet5', graph= tf.get_default_graph())
            writer.close()
            
            # just saver the last w and b
            saver.save(sess, 'model/model.ckpt', global_step=step)
            print("trainning is over!")
        elif argv[1] == '0':
            # 0 test
            ckpt = tf.train.get_checkpoint_state('model/') 
            print(ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("test accuracy %g" %(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})))
        else:
            img = Image.open(argv[1]).convert('L')
            # resize
            if img.size[0] != 28 or img.size[1] != 28:
                img = img.resize((28, 28))
            # store to arr
            arr = []
            for i in range(28):
                for j in range(28):
                    pixel = 1.0 - float(img.getpixel((j, i)))/255.0
                    arr.append(pixel)
            arr1 = np.array(arr).reshape((1, 784))
            
            gray = np.array(arr1).reshape((28, 28)) 
            x0=0; x1=0; y0=0; y1=0
            while np.sum(gray[0]) == 0:
                gray = gray[1:]
                y0 +=1

            while np.sum(gray[:,0]) == 0:
                gray = np.delete(gray,0,1)
                x0 +=1

            while np.sum(gray[-1]) == 0:
                gray = gray[:-1]
                y1 +=1

            while np.sum(gray[:,-1]) == 0:
                gray = np.delete(gray,-1,1)
                x1 +=1

            rows,cols = gray.shape
            region = (x0,y0,28-x1,28-y1)
            crop_img = img.crop(region) 
            if rows>cols:
                factor = 20.0/rows
                rows = 20
                cols = int(round(cols*factor))
            else:
                factor = 20.0/cols
                cols = 20
                rows = int(round(rows*factor))
            num_img = crop_img.resize((cols,rows))
            index_i = int((28 - rows)/2)
            index_j = int((28 - cols)/2)
            out = []
            for i in range(28):
                for j in range(28):
                    if index_i-1 < i < 28- index_i-1 and index_j-1 < j < 28-index_j-1:
                        #pixel = arr_num[i - index_i, j - index_j]
                        pixel = (num_img.getpixel((j-index_j, i-index_i)))/255
                    else:
                        pixel = 1.0
                    out.append(pixel)
            
            ckpt = tf.train.get_checkpoint_state('model/')
            print(ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                out = sess.run(y_conv, feed_dict={x: arr1, keep_prob: 1.0})
                out_num = tf.argmax(out, 1)
                print("your input number is: %d" %(sess.run(out_num)))
                
def main(argv):
    if os.path.exists("model"):
        pass
    else: 
        os.mkdir("model")
    run(argv)

if __name__ == '__main__':
     main(sys.argv)