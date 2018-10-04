
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


Ww = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32, name="weights")
bb = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32, name="biases")

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weigths:",sess.run(Ww))
    print("biases:",sess.run(bb))

