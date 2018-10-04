
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
print("test.shape")
print(mnist.test.images.shape)
print(mnist.test.labels.shape)


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


# ## 打印保存的参数值

# In[10]:


import os
from tensorflow.python import pywrap_tensorflow

current_path = os.getcwd()
model_dir = os.path.join(current_path, 'my_cnn')
checkpoint_path = os.path.join(model_dir,'save_net.ckpt') # 保存的ckpt文件名，不一定是这个
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key)) # 打印变量的值，对我们查找问题没啥影响，打印出来反而影响找问题


# 由于使用了tf.train.AdamOptimizer()来更新梯度，所以在保存检查点的时候如果不指定则是全局保存，把优化的变量“w_out/Adam”这种命名规则的变量也一并保存了，自然在恢复的时候就会出现找不到XX变量。解决办法，在声明 saver = tf.train.Saver()的时候带上参数，即需要保存的变量--------"https://blog.csdn.net/qq_36810544/article/details/81456182"

# ## 载入参数

# In[11]:


include = ['W_conv1','b_conv1','W_conv2','b_conv2','W_fc1','b_fc1','W_fc2','b_fc2']

variables_to_restore = tf.contrib.slim.get_variables_to_restore(include=include)
saver = tf.train.Saver(variables_to_restore)

print("使用载入的模型参数，测试训练集预测准确度：")
with tf.Session() as sess:
    saver.restore(sess, "my_cnn/save_net.ckpt")
    print(compute_accuracy(
            mnist.test.images, mnist.test.labels))

