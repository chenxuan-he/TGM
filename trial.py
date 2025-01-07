import numpy as np
from functions import permutation_test

# Generate datasets X and Y
def generate_data(n, m, p):
    # Generate X and Y from the same Gaussian distribution
    X = np.random.normal(0, 1, size=(n, p))
    Y = np.random.normal(0, 1, size=(m, p))
    return X, Y

# Example usage
n, m, p = 100, 100, 50  # Sample sizes and dimension
nsim = 100  # Number of simulations

X, Y = generate_data(n, m, p)
original_statistic, shuffled_statistics, p_value = permutation_test(X, Y, nsim)

print(f"Original Statistic: {original_statistic}")
print(f"P-value: {p_value}")

