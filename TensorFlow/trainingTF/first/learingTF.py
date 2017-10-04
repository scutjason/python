import tensorflow as tf

'''
# TensorBoard && scope
with tf.name_scope("Scope_A"):
    a = tf.add(1, 2, name = "A_add")
    b = tf.multiply(a, 3, name="A_mul")
    
with tf.name_scope("Scope_B"):
    c = tf.add(4, 5, name = "B_add")
    d = tf.multiply(c, 6, name = "B_mul")
    
e = tf.add(b, d, name = "output")

writer = tf.summary.FileWriter('./name_scope_1', graph= tf.get_default_graph())
writer.close()
'''

'''
# now, we create a complex scope
graph = tf.Graph()
with graph.as_default():
    in_1 = tf.placeholder(tf.float32, shape=[], name='input_a')
    in_2 = tf.placeholder(tf.float32, shape=[], name='input_b')
    const= tf.constant(3, dtype =tf.float32, name ='static_value')
    
    with tf.name_scope("Transformation"):
        with tf.name_scope("A"):
            A_mul = tf.multiply(in_1, const)
            A_out = tf.subtract(A_mul, in_1)
            
        with tf.name_scope("B"):
            B_mul = tf.multiply(in_2, const)
            B_out = tf.subtract(B_mul, in_2)
            
        with tf.name_scope("C"):
            C_div = tf.div(A_out, B_out)
            C_out = tf.add(C_div, const)
            
        with tf.name_scope("D"):
            D_div = tf.div(B_out, A_out)
            D_out = tf.add(D_div, const)
            
    out = tf.maximum(C_out, D_out)
    
writer = tf.summary.FileWriter('./name_scope_2', graph = graph)
writer.close()
'''

# L2 regularization 
batch_size = 8
def get_weight(shape, lamb):
    weight = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    
    # L2 add to list of "loss"
    tf.add_to_collection("loss", tf.contrib.layers.l2_regularizer(lamb)(weight))
    return weight
    
x = tf.placeholder("float", [None, 20])
y = tf.placeholder("float", [None, 8])

# net is  all_connect
# input is 20 cells
# hide1 is 10 cells
# hide2 is 10 cells
# output is 8 cells
net = [20, 10, 10, 8]
layers = len(net)

cur_net = x
in_net = net[0]

for i in range(1,layers):
    out_net = net[i]
    weights = get_weight([in_net, out_net], 0.1)
    biases = tf.random_normal([net[i]], dtype=tf.float32)
    cur_net = tf.nn.relu(tf.matmul(cur_net, weights) + biases)
    in_net = net[i]

# Standard Deviation  loss = tf.reduce_mean(tf.square(y-y_) + tf.contrib.layers.l2_regularizer(lambda)(w))
mse_loss = tf.reduce_mean(tf.square(cur_net - y))
tf.add_to_collection("loss", mse_loss)

# get_collection : get all the list of "loss"
# add_n : all the element added, and return the List like get_collection
loss = tf.add_n(tf.get_collection("loss"))
print(loss)



















