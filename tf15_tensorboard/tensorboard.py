
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ## histogram中监测各层的Weights, biases, outputs

# In[2]:


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name="W")
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('Weights'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name="b")
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights),biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)    
        return outputs


# ## 定义输入输出，添加层

# In[3]:


# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1],name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1],name="y_input")
    

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1,activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2,activation_function=None)


# ## scalar中监测loss的变化

# In[4]:


# the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)


# ## 训练

# In[5]:


with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
# important step
sess = tf.Session()

# 合并所有的summary到SummaryWriter中
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

# tf.train.SummaryWriter soon be deprecated, use following
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('logs/', sess.graph)
else: # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)


# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)



for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged,
                          feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)


# In[6]:


print("3 layers: input output relu")
print("x:", x_data.shape)
print("W1: (1,10)")
print("temp=W1*x+b1: (300,10)")
print("W2: (10,1)")
print("y=temp*W2")
print("y:", y_data.shape)


# 命令行中执行
# ```tensorboard --logdir='logs/'```
# 
# http://localhost:6006 中显示结果
