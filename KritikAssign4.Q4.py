import matplotlib.pyplot as plt
import numpy as np


def gradient_descent(f, learning_rate, initial_point, max_steps=10000, tolerance=1e-10):
    """
    Function to perform gradient descent to find the local minimum of a function f.

    Parameters:
    f (function): The function to minimize.
    learning_rate (float): The learning rate (step size).
    initial_point (float): The starting point for the gradient descent.
    max_steps (int): The maximum number of steps to take.
    tolerance (float): The tolerance for stopping the algorithm.

    Returns:
    tuple: The last x_n and f(x_n) values, rounded to three decimal places.
    """

    def deriv(f, base_point):
        """
        Estimate the derivative of the function f at base_point using the symmetric approximation.

        Parameters:
        f (function): The function whose derivative is to be estimated.
        base_point (float): The point at which to estimate the derivative.

        Returns:
        float: The estimated derivative at base_point.
        """
        return (f(base_point + 10 ** (-10)) - f(base_point - 10 ** (-10))) / (2 * 10 ** (-10))

    # Lists to store the x_n and f(x_n) values
    x_coords = [initial_point]
    y_coords = [f(initial_point)]

    # Perform gradient descent
    for _ in range(max_steps):
        current_point = x_coords[-1]
        current_derivative = deriv(f, current_point)

        # Check if the derivative is close enough to zero
        if abs(current_derivative) < tolerance:
            break

        # Update the point based on the learning rate and derivative
        next_point = current_point - learning_rate * current_derivative
        x_coords.append(next_point)
        y_coords.append(f(next_point))

    # Plotting the function and the sequence of points
    plot_range = np.linspace(min(x_coords) - 0.5, max(x_coords) + 0.5, 10000)
    function_range = [f(i) for i in plot_range]

    plt.plot(plot_range, function_range, label='Function f(x)')
    plt.plot(x_coords, y_coords, marker='o', label='Gradient Descent Steps')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title('Gradient Descent')
    plt.show()

    return round(x_coords[-1], 3), round(y_coords[-1], 3)


# Example usage:
# Define the function f(x) = x^2
def f(x):
    return x ** 2


# Perform gradient descent with a learning rate of 0.8 and initial point of 1
result_f = gradient_descent(f, learning_rate=0.8, initial_point=1)
print(f"Local minimum of f(x) = x^2 at x = {result_f[0]}, f(x) = {result_f[1]}")


# Define the function f(x) = x^4 - 2x^2
def g(x):
    return x ** 4 - 2 * x ** 2


# Perform gradient descent with a learning rate of 0.1 and initial point of 1
result_g_1 = gradient_descent(g, learning_rate=0.1, initial_point=1)
print(f"Local minimum of f(x) = x^4 - 2x^2 at x = {result_g_1[0]}, f(x) = {result_g_1[1]}")

# Perform gradient descent with a learning rate of 0.1 and initial point of -1
result_g_2 = gradient_descent(g, learning_rate=0.1, initial_point=-1)
print(f"Local minimum of f(x) = x^4 - 2x^2 at x = {result_g_2[0]}, f(x) = {result_g_2[1]}")

# Perform gradient descent with a learning rate of 0.1 and initial point of 0
result_g_3 = gradient_descent(g, learning_rate=0.1, initial_point=0)
print(f"Local minimum of f(x) = x^4 - 2x^2 at x = {result_g_3[0]}, f(x) = {result_g_3[1]}")


# Define the symmetrized version of x^x
def funny_function(x):
    if x > 0:
        return x ** x
    elif x == 0:
        return 1
    else:
        return abs(x) ** abs(x)


# Perform gradient descent with a learning rate of 0.01 and initial point of 0.5
result_funny_1 = gradient_descent(funny_function, learning_rate=0.01, initial_point=0.5)
print(f"Local minimum of funny_function at x = {result_funny_1[0]}, f(x) = {result_funny_1[1]}")

# Perform gradient descent with a learning rate of 0.01 and initial point of -0.5
result_funny_2 = gradient_descent(funny_function, learning_rate=0.01, initial_point=-0.5)
print(f"Local minimum of funny_function at x = {result_funny_2[0]}, f(x) = {result_funny_2[1]}")


# Define the function f(x) = |x|
def absolute_value_function(x):
    return abs(x)


# Perform gradient descent with a learning rate of 0.1 and initial point of 1
result_abs_1 = gradient_descent(absolute_value_function, learning_rate=0.1, initial_point=1)
print(f"Local minimum of |x| at x = {result_abs_1[0]}, f(x) = {result_abs_1[1]}")

# Perform gradient descent with a learning rate of 0.1 and initial point of -1
result_abs_2 = gradient_descent(absolute_value_function, learning_rate=0.1, initial_point=-1)
print(f"Local minimum of |x| at x = {result_abs_2[0]}, f(x) = {result_abs_2[1]}")

# Explanation:
# Gradient descent does not work well for the function |x| because |x| is not differentiable at x = 0.
# The derivative of |x| is -1 for x < 0 and 1 for x > 0. At x = 0, the derivative is undefined.
# As a result, gradient descent will oscillate around x = 0 without converging to it.