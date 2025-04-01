import math

def arctan_approximation(x):
    if x<0 or x>1:
        return "Error!"
    a = 0
    n = 0
    error_bound = x

    while error_bound > 0.0001:
        term = ((-1)**n)*(x**(2*n+1)) / (2*n+1)
        a += term
        n += 1
        error_bound = (x**(2*n+1)) / (2*n+1)
    return a, n, error_bound

test_input = [-1, 0, 0.25, 0.5, 0.75, 1]
test_output = [arctan_approximation(x) for x in test_input]
print(str(test_output))


