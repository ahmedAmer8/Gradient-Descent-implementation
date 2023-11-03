# imports
import numpy as np
import matplotlib.pyplot as plt

# weight is [2.89114079]
# bias is [2.58109277]

# generate random data-set
np.random.seed(0)
X = np.random.rand(100, 1)
Y = 2 + 3 * X + np.random.rand(100, 1)



def gradient_descent(x, y, learning_rate = 0.001, iterations = 100000, stopping_threshold = 1e-6):
    # initialize parameters
    current_weight = 0.1
    current_bias = 0.01
    iterations = iterations
    learning_rate = learning_rate
    stopping_threshold = stopping_threshold
    m = x.shape[0]
    for i in range(iterations):
        # updata the parameters
        # new weight = current_weight - alpha * dJ/dw
        y_predicted = (current_weight * x) + current_bias
        # dJdw = -(2/m) * np.dot(x.T, (y-y_predicted))
        dJdw = 1 / m * np.dot(x.T, (y_predicted - y))             # we use vectorization (np.dot()) for speed purpose
        temp_weight = current_weight - (learning_rate * dJdw)
        # if i < 13:
        #     print(temp_weight)
        # new bias = current_bias - alpha * dJ/db
        dJdb = 1/m * sum(y_predicted - y)
        # dJdb = -(2/m) * sum(y-y_predicted)
        temp_bias = current_bias - (learning_rate * dJdb)
        # if i < 13:
        #     print(temp_bias)
        if current_weight and abs(current_weight - temp_weight) <= stopping_threshold:
            break
        # assign temp weight and temp bias to bias and weight
        current_weight = temp_weight
        current_bias = temp_bias
    return current_weight, current_bias


w, b = gradient_descent(X, Y)
Y_pred = (w * X) + b
print(f"weight = {w} \nbias = {b}")
plt.scatter(X, Y, color= 'red')
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)],linestyle='dashed')
plt.ylabel("Y", fontsize= 20)
plt.xlabel("X", fontsize= 20)
plt.title("Gradient Descent", fontsize= 20)
plt.show()
