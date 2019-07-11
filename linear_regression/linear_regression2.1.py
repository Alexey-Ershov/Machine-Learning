import numpy as np
import tensorflow as tf


n_samples = 1000
batch_size = 100
num_steps = 20000
display_step = 100

np.random.seed(101)
tf.set_random_seed(101)

X_data = np.random.uniform(1, 10, (n_samples, 1))
y_data = 2*X_data + 1 + np.random.normal(0, 2, (n_samples, 1))

X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope('linear_regression'):
    k = tf.Variable(tf.random_normal((1, 1)), name="k")
    b = tf.Variable(tf.zeros((1,)), name="b")

learning_rate = 0.01
training_epochs = 20000

y_pred = tf.add(tf.multiply(X, k), b)
loss = tf.reduce_sum(tf.pow(y_pred - y, 2)) / (2*len(X_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_steps):
        indices = np.random.choice(n_samples, batch_size)
        X_batch = X_data[indices]
        y_batch = y_data[indices]

        opt_val, loss_val, k_val, b_val = \
                sess.run([optimizer, loss, k, b],
                         feed_dict={X: X_batch, y: y_batch})

        if (epoch+1) % display_step == 0:
            print("Epoch {}: loss = {:.8f}, k = {:.4f}, b = {:.4f}".
                    format(epoch + 1, loss_val, k_val[0][0], b_val[0]))
    
#     training_cost = sess.run(loss, feed_dict={X: X_data, y: y_data})
#     weight = sess.run(k)
#     bias = sess.run(b)

# # Calculating the predictions
# predictions = weight*X_data + bias
# print("predictions =", predictions,
#       "Training loss =", training_cost,
#       "Weight =", weight,
#       "bias =", bias, '\n')
