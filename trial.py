import numpy as np
from functions import U_L_MT_ED

# Generate datasets X and Y
def generate_data(n, m, p):
    # Generate X and Y from the same Gaussian distribution
    X = np.random.normal(0, 1, size=(n, p))
    Y = np.random.normal(0, 1, size=(m, p))
    return X, Y


# Permutation test
def permutation_test(X, Y, nsim=100):
    n, p = X.shape
    m = Y.shape[0]
    # Calculate the original statistic
    original_statistic = U_L_MT_ED(X, Y)
    # Combine X and Y
    combined = np.vstack((X, Y))
    # Perform permutations
    shuffled_statistics = []
    for _ in range(nsim):
        np.random.shuffle(combined)
        X_prime = combined[:n, :]
        Y_prime = combined[n:, :]
        shuffled_statistic = U_L_MT_ED(X_prime, Y_prime)
        shuffled_statistics.append(shuffled_statistic)
    # Calculate p-value
    shuffled_statistics = np.array(shuffled_statistics)
    p_value = np.mean(shuffled_statistics >= original_statistic)
    return original_statistic, shuffled_statistics, p_value

# Example usage
n, m, p = 100, 100, 50  # Sample sizes and dimension
nsim = 100  # Number of simulations

X, Y = generate_data(n, m, p)
original_statistic, shuffled_statistics, p_value = permutation_test(X, Y, nsim)

print(f"Original Statistic: {original_statistic}")
print(f"P-value: {p_value}")

