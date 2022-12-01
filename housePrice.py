import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import matplotlib.pyplot as plt

learning_rate = 0.02
num_epochs = 10000

D = np.matrix(pd.read_excel("housePriceData.xlsx", header=None).values)

#input columns
X_data = D[1:, 2:5].transpose()

#output column
y_data = D[1:, 7].transpose()


n = 3 # features
n_samples = y_data.size

# Define data placeholders
x = tf.placeholder(tf.float32, shape=(n, None))
y = tf.placeholder(tf.float32, shape=(1, None))

# Define trainable variables
A = tf.get_variable("A", shape=(1, n))
b = tf.get_variable("b", shape=())

# Define model output
y_predicted = tf.matmul(A, x) + b

# Define the loss function
L = tf.reduce_sum((y_predicted - y)**2)

# Define optimizer object
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(L)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(L)

# Create a session and initialize variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# Main optimization loop
for t in range(num_epochs):
    _, current_loss, current_A, current_b = session.run([optimizer, L, A, b], feed_dict={
        x: X_data,
        y: y_data
    })
    print("t = %g, loss = %g, A = %s, b = %g" % (t, current_loss, str(current_A), current_b))


theta = current_A[0]
bias = current_b
X_data = np.matrix(X_data)
y_data = np.matrix(y_data)


error_sum = 0
for i in range(n_samples):
    temp = [X_data.item(0,i), X_data.item(1,i), X_data.item(2,i)]
    predict = np.dot(temp,theta) + bias
    actual = y_data.item(i)
    error = (actual-predict)/actual * 100.0
    error_sum += abs(error)
    print(f"actual: {actual} , prediction: {predict}, error: = {error}")

print(f"avg error : {int(error_sum/n_samples)}%")