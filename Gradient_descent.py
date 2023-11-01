import numpy as np
import matplotlib.pyplot as plt
# Data
X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
              55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
              45.41973014, 54.35163488, 44.1640495, 58.16847072, 56.72720806,
              48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
Y = np.array([31.70700585, 68.77759598, 62.5623823, 71.54663223, 87.23092513,
              78.21151827, 79.64197305, 59.17148932, 75.3312423, 71.30087989,
              55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
              60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])

def gradient_descent(x, y, learning_rate = 0.0001, iterations = 100000, stopping_threshold = 1e-6):
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
        # dJ/dw = 1/m * sum((y_predicted - y) * x)
        y_predicted = (current_weight * x) + current_bias
        dJdw = -(2/m) * sum(x * (y-y_predicted))
        temp_weight = current_weight - (learning_rate * dJdw)
        # if i < 13:
        #     print(temp_weight)
        # new bias = current_bias - alpha * dJ/db
        # dJ/db = 1/m * sum(y_predicted - y)
        dJdb = -(2/m) * sum(y-y_predicted)
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
print(w, b)
plt.scatter(X, Y, color= 'red')
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)],linestyle='dashed')
plt.ylabel("Y", fontsize= 20)
plt.xlabel("X", fontsize= 20)
plt.title("Gradient Descent", fontsize= 20)
plt.show()