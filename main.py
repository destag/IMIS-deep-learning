import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from PIL import Image
from sklearn.preprocessing import normalize
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

#uczenie
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#interactive session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start =time.time()
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(200)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Skuteczność {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
print("Uczenie trwało {}".format(time.time()-start))
#print(mnist.test.images[0], mnist.test.labels[0])

im = Image.open("C:/Users/SzymonG/Desktop/dwa.jpg")
x = tf.placeholder([1, 784])
y_ = tf.placeholder([1, 10])
norm2 = normalize(np.array(im.getdata())[:,np.newaxis], axis=0).ravel()
print(norm2)
print("powinno być 2, a jest {}".format(sess.run(feed_dict={x: [norm2], y_: [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]})))
