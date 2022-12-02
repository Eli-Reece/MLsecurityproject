import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import matplotlib.pyplot as plt

# First we load the entire CSV file into an m x 3
D = np.matrix(pd.read_csv("TSLA.csv", header=None).values)

# We extract all rows and the first 2 columns into X_data
# Then we flip it


in1 = np.asarray(D[1:, 1])
in2 = np.asarray(D[1:, 2])
in3 = np.asarray(D[1:, 3])
in4 = np.asarray(D[1:, 5])
in5 = np.asarray(D[1:, 6])

out = np.asarray(D[1:, 4])

in_columns = np.column_stack((in1, in2, in3, in4, in5))

X_data = np.asarray(in_columns).transpose()

y_data = np.asarray(out).transpose()

n = in_columns[0].size
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
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000000001).minimize(L)
optimizer = tf.train.AdamOptimizer(learning_rate=0.5).minimize(L)

# Create a session and initialize variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# Main optimization loop
for t in range(1000):
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
    temp = []
    for j in range(n):
        temp.append(float(X_data.item(j,i)))

    predict = np.dot(temp,theta) + bias
    actual = float(y_data.item(i))
    error = (actual-predict)/actual * 100.0
    error_sum += abs(error)
    print(f"actual: {actual} , prediction: {predict}, error: = {error}")

print(f"avg error : {int(error_sum/n_samples)}%")