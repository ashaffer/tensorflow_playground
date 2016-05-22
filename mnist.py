#
# Imports
#

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#
# Parameters
#

x = tf.placeholder(tf.float32, [None, 784])  # Image dataset [][784]
W = tf.Variable(tf.zeros([784, 10]))         # Weights matrix [784]10] (784 pixels mapped to each output digit)
b = tf.Variable(tf.zeros([10]))              # Bias [10]. Prior probability of each digit?

#
# Model
#

y = tf.nn.softmax(tf.matmul(x, W) + b) # xW + b is how we classify an image or set of images
                                       # softmax is a logistic function (sigmoid curve)
                                       # normalized to be a probability distribution
                                       # (values between [0,1] that sum to 1)

#
# Loss function (cross-entropy)
#

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#
# Training - Gradient descent back-propagation
#

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#
# Initialization
#

init = tf.initialize_all_variables()

#
# Read in our training data
#

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)  # Grab the images with the classifications specified as 'one hot' vectors.
                                                                # That is, [0, 1, 0, 0, 0, 0, 0,  0, 0, 0] would be an output vector for the number 1.

#
# Run
#

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)              # Retrieve 100 samples from our data set of images (xs) and ys (classifications)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})    # Swap our random samples into the x and y placeholders, respectively

#
# Evaluation
#

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))    # A prediction is accurate if the second dimension of y (our output) equals the second dimension of y_ (the correct output)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # Cast the boolean results of the previous line to floats (True -> 1.0, False -> 0.0) and find the mean over all the images in the test set

#
# Output
#

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
