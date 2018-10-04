
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# # load data

# In[2]:


# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[3]:


print("train.shape")
print(mnist.train.images.shape)
print(mnist.train.labels.shape)


# In[4]:


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob:1}) # y_pre 是一个1*10的概率向量
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys,keep_prob:1}) # 百分比
    return result


# ## 定义变量

# In[5]:


def weight_variable(shape,name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial,name=name)


# In[6]:


def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial,name=name)


# ## 定义层

# In[7]:


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# "SAME"padding，抽取出的和原图大小相同，外围用0来填充再做卷积


# In[8]:


# pooling 一般不需要 padding？padding=0,即valid poolong
def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


# ## 构造网络
# 两个（卷积+最大池化），两个全联接层

# In[9]:


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 784]) # 28*28
ys = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(xs,[-1,28,28,1]) # [n_samples, 28,28,1] -1 具体是多少由导入数据决定（多少张图片）
print(x_image.shape) 
    
def my_cnn():
    ## conv1 layer ##
    W_conv1 = weight_variable([5,5,1,32],name="W_conv1") # patch: 5*5, in size 1(通道数，image的厚度), out size 32（feature_map数量，一个卷积核生成一个feature_map）
    b_conv1 = bias_variable([32],name="b_conv1")
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)# output size 28*28*32
    h_pool1 = max_pooling_2x2(h_conv1) # output size 14*14*32

    ## conv2 layer ##
    W_conv2 = weight_variable([5,5,32,64],name="W_conv2") # patch: 5*5, in size 32，out size 64
    b_conv2 = bias_variable([64],name="b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)# output size 14*14*64
    h_pool2 = max_pooling_2x2(h_conv2) # output size 7*7*64

    ## func1 layer ##
    W_fc1 = weight_variable([7*7*64,1024],name="W_fc1")
    b_fc1 = bias_variable([1024],name="b_fc1")

    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64]) # [n_samples,7,7,64] ->> [n_samples,7*7*64]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    ## func2 layer ##
    W_fc2 = weight_variable([1024,10],name="W_fc2")
    b_fc2 = bias_variable([10],name="b_fc2")
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    var_dict = {'W_conv1': W_conv1, 
                'b_conv1': b_conv1, 
                'W_conv2': W_conv2, 
                'b_conv2': b_conv2, 
                'W_fc1': W_fc1, 
                'b_fc1': b_fc1,
                'W_fc2': W_fc2,
                'b_fc2': b_fc2}
    return prediction, var_dict


# ## loss

# In[10]:


prediction, var_dict = my_cnn()

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss


# ## train

# In[11]:


train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    

sess = tf.Session()

# important step
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver(var_dict)

print("准确率的提升过程...")
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))


# 最终达到97%的准确度
# - 0.14
# - 0.762
# - 0.845
# - 0.901
# - 0.915
# - 0.921
# - 0.937
# - 0.936
# - 0.94
# - 0.952
# - 0.952
# - 0.951
# - 0.952
# - 0.958
# - 0.96
# - 0.962
# - 0.961
# - 0.962
# - 0.97
# - 0.964

# In[12]:


# 只保存第1000次的参数
save_path = saver.save(sess,"my_cnn/save_net.ckpt")
print("Save to path:", save_path)

