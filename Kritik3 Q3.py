import math



def central_difference(f, c, delta_x=1e-8):
    return (f(c + delta_x) - f(c - delta_x)) / (2 * delta_x)



def linear_approximation(f, df, c, x):
    return f(c) + df * (x - c)



def find_x1_x2(f, c, E, max_iter=100000, delta=1e-5):

    df_c = central_difference(f, c)


    def L(x):
        return linear_approximation(f, df_c, c, x)

    x1, x2 = None, None
    for i in range(1, max_iter):
        x_left = c - i * delta
        x_right = c + i * delta


        error_left = abs(f(x_left) - L(x_left))
        error_right = abs(f(x_right) - L(x_right))


        if error_left >= E - 1e-5 and error_left <= E + 1e-5 and not x1:
            x1 = x_left

        if error_right >= E - 1e-5 and error_right <= E + 1e-5 and not x2:
            x2 = x_right


        if x1 and x2:
            break

    return x1, x2 if x1 and x2 else "No solution found within range."

def f1(x):
    return x ** 2

c1 = 1
E1 = 0.1
x1_1, x2_1 = find_x1_x2(f1, c1, E1)
print(f"Test case 1: x1 = {x1_1}, x2 = {x2_1}")


def f2(x):
    return math.sin(x)

c2 = math.pi / 4
E2 = 0.05
x1_2, x2_2 = find_x1_x2(f2, c2, E2)
print(f"Test case 2: x1 = {x1_2}, x2 = {x2_2}")


def f3(x):
    return math.exp(x)


c3 = 0
E3 = 0.01
x1_3, x2_3 = find_x1_x2(f3, c3, E3)
print(f"Test case 3: x1 = {x1_3}, x2 = {x2_3}")


def dR_dt(R):
    return (1 / tau) * (1 - 1.15 * R - 0.9 * np.exp(-k * tau * R))

# Euler's method implementation

