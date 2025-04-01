import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gradient_descent(x0, y0, f, grad_f, alpha, num_iterations):
    """
    Performs gradient descent to find the local minimum of a function.

    Parameters:
    x0, y0: Initial point for the descent.
    f: Function to minimize.
    grad_f: Gradient function of f.
    alpha: Learning rate.
    num_iterations: Number of iterations.

    Returns:
    (x, y): Coordinates of the final point after gradient descent.
    """
    x, y = x0, y0  # Initialize variables

    for _ in range(num_iterations):
        grad_x, grad_y = grad_f(x, y)  # Compute gradient at (x, y)
        x -= alpha * grad_x  # Update x in the opposite direction of gradient
        y -= alpha * grad_y  # Update y in the opposite direction of gradient

    return x, y


# Function f1(x, y) = x^2 + y^2
def fun_1(x, y):
    return x ** 2 + y ** 2


# Gradient of f1

def grad_f_1(x, y):
    grad_x = 2 * x
    grad_y = 2 * y
    return grad_x, grad_y


# Apply gradient descent on f1
x1, y1 = gradient_descent(0.1, 0.1, fun_1, grad_f_1, 0.1, 10)
x2, y2 = gradient_descent(-1, 1, fun_1, grad_f_1, 0.01, 100)
print(
    f"For f1: Gradient descent converges to ({x1:.5f}, {y1:.5f}) and ({x2:.5f}, {y2:.5f}), meaning it reaches the global minimum (0,0).")


# Function f2(x, y)
def fun_2(x, y):
    return 1 - np.exp(-x ** 2 - (y - 2) ** 2) - 2 * np.exp(-x ** 2 - (y + 2) ** 2)


# Gradient of f2
def grad_f_2(x, y):
    grad_x = (2 * x * np.exp(-x ** 2 - (y - 2) ** 2) + 4 * x * np.exp(-x ** 2 - (y + 2) ** 2))
    grad_y = (2 * (y - 2) * np.exp(-x ** 2 - (y - 2) ** 2) + 4 * (y + 2) * np.exp(-x ** 2 - (y + 2) ** 2))
    return grad_x, grad_y


# Apply gradient descent on f2
x3, y3 = gradient_descent(0, 1, fun_2, grad_f_2, 0.01, 10000)
x4, y4 = gradient_descent(0, -1, fun_2, grad_f_2, 0.01, 10000)
print(
    f"For f2: Gradient descent converges to ({x3:.5f}, {y3:.5f}) and ({x4:.5f}, {y4:.5f}), showing it reaches different local minima depending on the starting point.")

# Plot f2(x, y)
X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(X, Y)
z = fun_2(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
plt.title("Surface plot of f2(x, y)")
plt.show()