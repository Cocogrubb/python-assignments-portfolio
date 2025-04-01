import numpy as np
from scipy.special import gamma


# Function to compute the mean of a list of numbers
def compute_mean(data):
    """
    Compute the mean (average) of the given list of data.

    Parameters:
    data (list of floats): The data points to compute the mean of.

    Returns:
    float: The mean of the data points.
    """
    return sum(data) / len(data)


# Function to compute the standard deviation of a list of numbers
def compute_standard_deviation(data):
    """
    Compute the standard deviation of the given list of data.

    Parameters:
    data (list of floats): The data points to compute the standard deviation of.

    Returns:
    float: The standard deviation of the data points.
    """
    # Calculate the mean
    mean = compute_mean(data)
    # Calculate the variance (sum of squared differences from the mean)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    # Return the square root of the variance to get the standard deviation
    return variance ** 0.5


# Function to compute the t0 value for hypothesis testing
def compute_t0(data, mu_0):
    """
    Compute the t0 value for hypothesis testing.

    Parameters:
    data (list of floats): The data points (sample).
    mu_0 (float): The hypothesized mean of the population.

    Returns:
    float: The computed t0 value.
    """
    # Sample mean
    x_bar = compute_mean(data)
    # Sample standard deviation
    s = compute_standard_deviation(data)
    # Sample size
    n = len(data)
    # Compute t0 using the formula
    t_0 = (x_bar - mu_0) / (s / (n ** 0.5))
    return t_0


# Function to compute the probability density function (PDF) of the t-distribution
def t_distribution_pdf(x, nu):
    """
    Compute the probability density of the t-distribution at a given point.

    Parameters:
    x (float): The point at which to evaluate the density.
    nu (int): The degrees of freedom of the t-distribution.

    Returns:
    float: The probability density at point x for the t-distribution with nu degrees of freedom.
    """
    coeff = gamma((nu + 1) / 2) / (np.sqrt(nu * np.pi) * gamma(nu / 2))
    return coeff * (1 + x ** 2 / nu) ** (-0.5 * (nu + 1))


# Function to find the t* value using numerical integration
def find_t_star(prob, nu, x_start=0, x_end=20, num_points=10000):
    """
    Find the t-value t* for a given cumulative probability and degrees of freedom.

    Parameters:
    prob (float): The cumulative probability (between 0 and 1).
    nu (int): The degrees of freedom of the t-distribution.
    x_start (float): The start point for numerical integration.
    x_end (float): The end point for numerical integration.
    num_points (int): The number of points to use in the numerical integration.

    Returns:
    float: The t-value t* such that the area between [-t*, t*] equals the given probability.
    """
    # Define the x values for integration
    x = np.linspace(x_start, x_end, num_points)
    # Apply the density function to the x values
    y = t_distribution_pdf(x, nu)
    # Cumulative sum for the area under the curve (numerical integration)
    cdf = np.cumsum(y) * (x[1] - x[0])
    # Find the t-value where the cumulative probability reaches half of the required probability
    target_half_prob = prob / 2
    index = np.where(cdf >= target_half_prob)[0][0]
    return x[index]


# Function to test the hypothesis: if t0 is within [-t*, t*]
def hypothesis_test(t_0, t_star):
    """
    Perform hypothesis testing: check if t0 is within [-t*, t*].

    Parameters:
    t_0 (float): The computed t0 value from the data.
    t_star (float): The critical t* value from the t-distribution.

    Returns:
    bool: True if t0 is within [-t*, t*], otherwise False.
    """
    return abs(t_0) <= t_star


# Main function to apply the t-test and interpret results
def apply_t_test(data, mu_0, confidence_level=0.95):
    """
    Apply the t-test for a given data sample, hypothesized mean, and confidence level.

    Parameters:
    data (list of floats): The sample data points.
    mu_0 (float): The hypothesized mean of the population.
    confidence_level (float): The desired confidence level (default is 0.95).

    Returns:
    None: Prints the results of the t-test and conclusion.
    """
    # Step 1: Compute the mean and standard deviation of the data
    mean = compute_mean(data)
    std_dev = compute_standard_deviation(data)

    # Step 2: Compute the t0 value for the sample data
    t_0 = compute_t0(data, mu_0)

    # Step 3: Find the critical t* value for the given confidence level
    nu = len(data) - 1  # Degrees of freedom (n-1)
    t_star = find_t_star(confidence_level, nu)

    # Step 4: Perform the hypothesis test
    reject_null = hypothesis_test(t_0, t_star)

    # Output the results
    print(f"Sample Mean: {mean}")
    print(f"Sample Standard Deviation: {std_dev}")
    print(f"Computed t0: {t_0}")
    print(f"Critical t* value: {t_star}")
    print(f"Reject null hypothesis: {reject_null}")

    if reject_null:
        print(
            "Conclusion: The new teaching technique significantly impacts the test scores (null hypothesis rejected).")
    else:
        print(
            "Conclusion: There is not enough evidence to suggest that the new teaching technique significantly impacts the test scores (fail to reject the null hypothesis).")


# Given data (test scores of 10 students)
data = [92.64, 79.00, 84.79, 97.41, 93.68, 65.23, 84.50, 73.49, 73.97, 79.11]
# Hypothesized population mean
mu_0 = 75

# Apply the t-test to the given data and print the results
apply_t_test(data, mu_0)

# Explanation of the code:
"""
1. **Mean and Standard Deviation Calculation:**
   - The `compute_mean` and `compute_standard_deviation` functions calculate the sample mean and sample standard deviation, respectively, using basic arithmetic operations.

2. **t0 Calculation:**
   - The `compute_t0` function calculates the t-statistic t0 for hypothesis testing using the formula:
     t0 = (x_bar - mu_0) / (s / (n ** 0.5)), where:
     - x_bar is the sample mean
     - mu_0 is the hypothesized population mean
     - s is the sample standard deviation
     - n is the sample size

3. **t* Calculation:**
   - The `find_t_star` function computes the critical t-value t* using numerical integration of the t-distribution's probability density function. This function ensures that the area under the curve from -t* to t* corresponds to the desired cumulative probability (0.95 for 95% confidence).

4. **Hypothesis Testing:**
   - The `hypothesis_test` function checks if the computed t0 lies within the range [-t*, t*]. If it does, we fail to reject the null hypothesis, suggesting no significant difference from the hypothesized mean. If not, we reject the null hypothesis, suggesting a significant difference.

5. **Applying the t-Test:**
   - The `apply_t_test` function ties all the steps together, computes the necessary values, and prints the result, including whether the null hypothesis should be rejected or not based on the computed t0 and t*.

### Conclusion:
After running the code, you will get the results of the t-test and the conclusion based on whether the computed t0 falls within the range of the critical t-values, allowing you to determine if the new teaching technique has a statistically significant impact on student test scores.
"""
