# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:23:52 2017

@author: Admin
"""


from __future__ import print_function
import tensorflow as tf
import os
#import matplotlib.pyplot as plt
from PIL import Image
import numpy as np  
from skimage import io
#import MNIST_data.input_data as input_data

tf.logging.set_verbosity(tf.logging.INFO)
def compute_accuracy(v_xs, v_ys):
    global regression#全局变量
    y_pre = sess.run(regression, feed_dict={xs: v_xs, keep_prob: 1})
    #accuracy=tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.square(tf.subtract(ys,y_pre)),reduction_indices=[1]), shape=[-1,1]))
    #accuracy=tf.reduce_mean(tf.square(tf.subtract(ys,y_pre)))
    
    accuracy=tf.sqrt(tf.reduce_sum(tf.square(ys-y_pre), 1))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0,stddev=0.01)#正态分布 stddev调整标准差 
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    

if __name__ == '__main__': 
# define placeholder for inputs to network

    xs = tf.placeholder(tf.float32, [None, 2048],name='xs') # 272*448，列确定，行不确定
    ys = tf.placeholder(tf.float32, [ None,784],name='ys')  
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    #x_image = tf.reshape(xs, [-1,256,256, 1])# 图像转化为一个四维张量，第一个参数代表样本数量，-1表示不定第二三参数代表图像尺寸，最后一个参数代表图像通道数


   ## fc1 layer ##
    W_fc1 = weight_variable([2048, 4096])
    b_fc1 = bias_variable([4096])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool1_flat = tf.reshape(xs, [-1, 2048])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
    #regression=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


   ## fc3 layer ##
    W_fc3 = weight_variable([4096, 4096])
    b_fc3 = bias_variable([4096])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool3_flat = tf.reshape(h_fc1_drop, [-1,4096])
    h_fc3 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc3) + b_fc3)
    #regression=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)


   ## fc6 layer ##
    W_fc6 = weight_variable([4096, 4096])
    b_fc6 = bias_variable([4096])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool6_flat = tf.reshape(h_fc3_drop, [-1,4096])
    h_fc6 = tf.nn.relu(tf.matmul(h_pool6_flat, W_fc6) + b_fc6)
    #regression=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)


   ## fc7 layer ##
    W_fc7 = weight_variable([4096, 4096])
    b_fc7 = bias_variable([4096])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool7_flat = tf.reshape(h_fc6_drop, [-1,4096])
    h_fc7 = tf.nn.relu(tf.matmul(h_pool7_flat, W_fc7) + b_fc7)
    #regression=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)


   ## fc8 layer ##
    W_fc8 = weight_variable([4096, 2048])
    b_fc8 = bias_variable([2048])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool8_flat = tf.reshape(h_fc7_drop, [-1,4096])
    h_fc8 = tf.nn.relu(tf.matmul(h_pool8_flat, W_fc8) + b_fc8)
    #regression=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc8_drop = tf.nn.dropout(h_fc8, keep_prob)

   ## fc4 layer ##
    W_fc4 = weight_variable([2048, 1024])
    b_fc4 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool4_flat = tf.reshape(h_fc8_drop, [-1,2048])
    h_fc4 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc4) + b_fc4)
    #regression=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)


   ## fc5 layer ##
    W_fc5 = weight_variable([1024,784])
    b_fc5 = bias_variable([784])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_pool5_flat = tf.reshape(h_fc4_drop, [-1,1024])
    #h_fc5= tf.nn.relu(tf.matmul(h_pool5_flat, W_fc5) + b_fc5)
    regression=tf.matmul(h_pool5_flat, W_fc5) + b_fc5
    #h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob)


 
    #regression
    #W_conv8 = weight_variable([1,7,32,1]) # patch 5x5, in size 32, out size 1
    #b_conv8 = bias_variable([1])#feature map 28*28*1
    #h_conv8=tf.nn.relu(tf.nn.conv2d(h_pool7, W_conv8, strides=[1, 1, 1, 1], padding='VALID') + b_conv8)#激活函数能够给神经网络加入一些非线性因素，更好地解决较为复杂的问题。
   # h_conv8=tf.nn.conv2d(h_pool7, W_conv8, strides=[1, 1, 1, 1], padding='VALID') + b_conv8
    #regression=tf.reshape(regression,[-1,784])
    
    
    # the error between prediction and real data 
    #mse = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(ys,regression)),reduction_indices=[1])) # loss,减法，平方。列相加，平均，开方
   # mse = tf.reduce_mean(tf.square(tf.subtract(ys,regression)))
    mse = tf.sqrt(tf.reduce_sum(tf.square(ys-regression), 1))
#reduce_sum() 就是求和,由于求和的对象是tensor,所以是沿着tensor的某些维度求和.reduction_indices是指沿tensor的哪些维度求和.0是行相加1是列相加
    train_step = tf.train.AdamOptimizer(0.0001).minimize(mse)




    imgs3 = os.listdir('D:\\SUN\\2000\\')
    imgNum3= len(imgs3)
    data3 = np.empty((2400,2048),dtype="float32")
    for i in range (imgNum3):
       img=io.imread('D:\\SUN\\2000\\'+imgs3[i],as_grey=True)
       arr = np.asarray(img,dtype="float32")
       arr = [y for x in arr for y in x]
       data3[i,:]=arr
    #train_s=data3/255
    train_s=2*data3/255-1



    imgs2 = os.listdir('D:\\SUN\\2000y\\')
    imgNum2= len(imgs2)
    data2 = np.empty((2400,784),dtype="float32")
    for i in range (imgNum2):
       img=io.imread('D:\\SUN\\2000y\\'+imgs2[i],as_grey=True)
       arr = np.asarray(img,dtype="float32")
       arr = [y for x in arr for y in x]
       data2[i,:]=arr
    #train_s=data3/255
    train_y=2*data2/255-1
  
 



    
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)   
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())  
        test_a=train_s[39*50:40*50]
        test_b=train_y[39*50:40*50]
        for i in range(5000):#训练2000次



           for j in range (39):#0到149
               temp_a=train_s[j*50:(j+1)*50]
        #   temp_a1=train_s[1*30:(1+1)*30]
               temp_b=train_y[j*50:(j+1)*50]
              
               

           #temp_b=train_y[0*30:(0+1)*30]  
           #temp_b1=train_y[1*30:(1+1)*30]  
               sess.run(train_step, feed_dict={xs:temp_a,ys:temp_b,keep_prob: 1})
            
               #feed_dict给使用placeholder创建出的tensor赋值，feed可使用一个值临时替换op的输出结果，可提供值给feed，feed只在调用它的方法内有效，方法结束，feed就会消失
              # print(compute_accuracy(temp_a,temp_b))
           #if i % 20 == 0:
           print(i)

           print(compute_accuracy(temp_a, temp_b))
           print(compute_accuracy(test_a, test_b))
            
           if i%20==0:
               
               y3 = sess.run(regression, feed_dict={xs:temp_a, keep_prob: 1})
               for k in range (50):
                   array3 = np.reshape(y3[k], newshape=[28,28])#原图第11854张               
                   y11=(array3)*255
                   max11=np.max(y11)
                   min11=np.min(y11)
                   y11=(y11-min11)/(max11-min11)*255
              # y1=(array+1)*255./2
                   y11= y11.astype(np.int)
               
                   #image11 = Image.fromarray(y11) 
                   io.imsave('D:\\SUN\\2000_train\\' + str(i) + str(k)+'.bmp', y11) 
               
               
               y = sess.run(regression, feed_dict={xs:test_a, keep_prob: 1})
               for k in range (50):
                   array = np.reshape(y[k], newshape=[28,28])#原图第11854张               
                   y1=(array)*255
                   max1=np.max(y1)
                   min1=np.min(y1)
                   y1=(y1-min1)/(max1-min1)*255
              # y1=(array+1)*255./2
                   y1= y1.astype(np.int)
                   
                   #image = Image.fromarray(y1) 
                   io.imsave('D:\\SUN\\2000_test\\' + str(i) + str(k)+'.bmp', y1)  
               

         