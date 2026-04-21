import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 10, 20)
y = 2.5 * x + 3 + np.random.randn(20)

m = 0
b = 0
lr = 0.01
epochs = 100

errors = []

for i in range(epochs):
    y_pred = m*x + b
    
    error = np.mean((y - y_pred)**2)
    errors.append(error)

    dm = (-2/len(x)) * np.sum(x*(y - y_pred))
    db = (-2/len(x)) * np.sum(y - y_pred)

    m = m - lr * dm
    b = b - lr * db

print("Final m:", m)
print("Final b:", b)

plt.plot(errors)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title("Error Reduction")
plt.show()

plt.scatter(x, y)
plt.plot(x, m*x + b)
plt.title("Best Fit Line")
plt.show()