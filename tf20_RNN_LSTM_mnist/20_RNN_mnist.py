
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# ## 导入数据

# In[2]:


mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


# ## 参数

# In[3]:


# hyperparameters
tf.set_random_seed(1)
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28   # MNIST data input (img shape: 28*28) 每次读入一行1*28
n_steps = 28    # time steps 纵向移动28次
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])


# ## 定义参数

# In[4]:


# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


# ## 构造网络
# 
# 不考虑batch_size时
# ```
# X = (28,28)
# 
# 对于X的每一行：
# 
# for X_line in X:
#     1.经隐藏层w=(28,128): X_line=(1,128)
#     每个step得到一个X_line(1,128)
#     2.经LSTM_Cell
#     每个step得到(1,128)的outpus
#     3.经隐藏层w=(128,10): 
#     每个step得到(1,10)results
#     
#     
#     
# 共进行28 steps，得到28个results(1,10)
# 
# 最终取，最后一个results = final_state[1]*w+b
# ```
# 考虑batch_size时，即增加深度。想象把128张“例图”重叠的放置在一起

# In[5]:


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################
    
    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    
    # cell
    ##########################################
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, m_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    
    outpus, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state,time_major=False)
    # 由于 steps 即时间序列长度为28，在X_in的次要维度，
    # 所以time_major=False，否则，28 steps 在X_in的第一个维度，
    # 则time_major=True

    # hidden layer for output as the final results
    ############################################
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # or
    # unpack to list [(batch, outputs)..] * steps
    # transpose 三维张量转制，102代表第一第二维度交换
    # outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out'] # shape = (128, 10)

    return results


# ## loss

# In[6]:


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)


# In[7]:


correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[8]:


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    x_test = mnist.test.images[:128]
    y_test = mnist.test.labels[:128]
    x_test = x_test.reshape([128, n_steps, n_inputs])
    print("测试集中前128个图的准确度:")
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs,
                                        y: batch_ys})
        if step % 50 == 0:
            print("每次读取128张图，第",step+1,"次")
            print(sess.run(accuracy, feed_dict={
                x:x_test,
                y:y_test
        }))
        step += 1

