
# coding: utf-8

# In[1]:


import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# # load data

# In[2]:


digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


# In[3]:


print(X.shape)
print(y.shape)


# In[4]:


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# In[5]:


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs}) # y_pre 是一个1*10的概率向量
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys}) # 百分比
    return result


# ## 定义输入输出，添加层

# In[6]:


# define placeholder for inputs to network

keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64]) # 8*8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)


# ## loss

# In[7]:


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
tf.summary.scalar('loss', cross_entropy)


# ## train

# In[8]:


train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)
    

sess = tf.Session()
merged = tf.summary.merge_all()
# summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

# important step
sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step,feed_dict={xs:X_train, ys:y_train, keep_prob:0.5}) # 每次随机dropout其中的0.5
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train,keep_prob:1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test,keep_prob:1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)


# ### dropout
# 
# 每层dropout一半之后，解决overfiting问题，在loss的曲线图，在测试集上的效果和训练集更为接近
