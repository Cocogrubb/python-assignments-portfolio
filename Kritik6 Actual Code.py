#3)
import numpy as np
import matplotlib.pyplot as plt

# Function to compute the normal density function value for given mean and variance at x
def normal_density(mean, variance, x):
    sigma = np.sqrt(variance)  # Compute standard deviation from variance
    return (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-((x - mean) ** 2) / (2 * variance))

# Generate x-values for plotting the normal density function
x_values = np.linspace(140, 200, 1000)
params = [(171, 7.12), (160, 10), (180, 5)]  # Different mean and variance pairs to compare

plt.figure(figsize=(10, 6))  # Set figure size for better visibility
for mean, variance in params:
    y_values = [normal_density(mean, variance, x) for x in x_values]  # Compute density values
    plt.plot(x_values, y_values, label=f"Mean={mean}, Variance={variance}")  # Plot the function
    plt.fill_between(x_values, y_values, alpha=0.3)  # Add fill between the curve

# Labeling the plot
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Normal Density Functions")
plt.legend()
plt.grid()
plt.show()

# Function to numerically integrate the normal density function using the trapezoidal rule
def integration(mean, variance, a, b, num_points=1000):
    x_vals = np.linspace(a, b, num_points)  # Generate x values for integration
    y_vals = [normal_density(mean, variance, x) for x in x_vals]  # Compute function values
    dx = (b - a) / (num_points - 1)  # Compute step size
    result = np.trapz(y_vals, x_vals)  # Use trapezoidal rule to approximate integral
    return result

# Compute probability for a randomly chosen man's height between 162cm and 190cm
mean_height = 171  # Given mean height
variance_height = 7.12  # Given variance
probability = integration(mean_height, variance_height, 162, 190)  # Compute probability

# Print the computed probability
print(f"The probability that a randomly chosen man has a height between 162cm and 190cm is {probability:.4f}")
print(" ")

#4)
import numpy as np

# 4(a) Uniform Distribution: Compute the expected value for a uniform distribution on the interval [a, b]
def uniform_expected_value(a, b):
    return (a + b) / 2

# 4(b) Exponential Distribution: Compute the expected value for an exponentially distributed variable with rate lambda
def exponential_expected_value(lambda_value):
    return 1 / lambda_value

# 4(c) Normal Distribution and Drug Dosage: Compute expected dosage based on height and variance
def normal_density(mean, variance, x):
    sigma = np.sqrt(variance)  # Compute standard deviation from variance
    return (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-((x - mean) ** 2) / (2 * variance))

def expected_dosage(mean, variance):
    # Compute E[D(X)] = 2.38 * (mu^2 + sigma^2)
    return 2.38 * (mean**2 + variance)

# Example Inputs

# --- Answer for 4(a) Uniform Distribution ---
a, b = 0, 10  # Example interval [a, b]
uniform_E = uniform_expected_value(a, b)
print(f"--- 4(a) Uniform Distribution ---")
print(f"The expected value for Uniform Distribution on [{a}, {b}] is: {uniform_E}\n")

# --- Answer for 4(b) Exponential Distribution ---
lambda_value = 1 / 50  # Example rate lambda = 1/50 for time between pandemics
exponential_E = exponential_expected_value(lambda_value)
print(f"--- 4(b) Exponential Distribution ---")
print(f"The expected value for Exponential Distribution with rate λ = 1/50 is: {exponential_E} years\n")

# --- Answer for 4(c) Normal Distribution and Drug Dosage ---
mean_height = 171  # Mean height in cm
variance_height = 7.1  # Variance in cm^2

# Compute expected drug dosage
expected_dosage_value = expected_dosage(mean_height, variance_height)
print(f"--- 4(c) Normal Distribution and Drug Dosage ---")
print(f"The expected dosage for a male with height normally distributed with mean {mean_height} cm and variance {variance_height} cm^2 is: {expected_dosage_value:.2f} units.")