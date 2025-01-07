import numpy as np

def Subtractr1(X, Y, r):
    # 1 <= r <= X.shape[0]
    m, p = X.shape
    n = Y.shape[0]
    Zctr1 = Y - np.ones((n, 1)) @ X[r, :].reshape(1, p)
    ZrNorm1 = np.sqrt(np.sum(Zctr1**2, axis=1)).reshape(-1, 1)
    ZrStd1 = Zctr1 / (ZrNorm1 @ np.ones((1, p)))

    Zctr2 = X - np.ones((m, 1)) @ X[r, :].reshape(1, p)
    ZrNorm2 = np.sqrt(np.sum(Zctr2**2, axis=1)).reshape(-1, 1)
    ZrStd2 = Zctr2 / (ZrNorm2 @ np.ones((1, p)))

    A = ZrStd1 @ ZrStd2.T
    A = np.clip(A, -1, 1)
    A = np.arccos(A)
    A[np.isnan(A)] = 0
    return 2 * np.sum(A) / (m * (m - 1) * n)

def Subtractr2(X, Y, r):
    # 1 <= r <= Y.shape[0]
    m, p = X.shape
    n = Y.shape[0]
    Zctr1 = X - np.ones((m, 1)) @ Y[r, :].reshape(1, p)
    ZrNorm1 = np.sqrt(np.sum(Zctr1**2, axis=1)).reshape(-1, 1)
    ZrStd1 = Zctr1 / (ZrNorm1 @ np.ones((1, p)))

    Zctr2 = Y - np.ones((n, 1)) @ Y[r, :].reshape(1, p)
    ZrNorm2 = np.sqrt(np.sum(Zctr2**2, axis=1)).reshape(-1, 1)
    ZrStd2 = Zctr2 / (ZrNorm2 @ np.ones((1, p)))

    A = ZrStd1 @ ZrStd2.T
    A = np.clip(A, -1, 1)
    A = np.arccos(A)
    A[np.isnan(A)] = 0
    return 2 * np.sum(A) / (n * (n - 1) * m)

def Subtractr3(Z, r):
    n, p = Z.shape
    Zctr = Z - np.ones((n, 1)) @ Z[r, :].reshape(1, p)
    Zctr = np.delete(Zctr, r, axis=0)

    ZrNorm = np.sqrt(np.sum(Zctr**2, axis=1)).reshape(-1, 1)
    ZrStd = Zctr / (ZrNorm @ np.ones((1, p)))

    A = ZrStd @ ZrStd.T
    A = np.clip(A, -1, 1)
    A = np.arccos(A)
    A[np.isnan(A)] = 0
    return np.sum(A) - np.sum(np.diag(A))

def Subtractr4(X, Y, r):
    # 1 <= r <= X.shape[0]
    m, p = X.shape
    n = Y.shape[0]
    Zctr1 = Y - np.ones((n, 1)) @ X[r, :].reshape(1, p)
    ZrNorm1 = np.sqrt(np.sum(Zctr1**2, axis=1)).reshape(-1, 1)
    ZrStd1 = Zctr1 / (ZrNorm1 @ np.ones((1, p)))

    A = ZrStd1 @ ZrStd1.T
    A = np.clip(A, -1, 1)
    A = np.arccos(A)
    A[np.isnan(A)] = 0
    return np.sum(A) / (m * n**2)

def Subtractr5(X, Y, r):
    # 1 <= r <= Y.shape[0]
    m, p = X.shape
    n = Y.shape[0]
    Zctr1 = X - np.ones((m, 1)) @ Y[r, :].reshape(1, p)
    ZrNorm1 = np.sqrt(np.sum(Zctr1**2, axis=1)).reshape(-1, 1)
    ZrStd1 = Zctr1 / (ZrNorm1 @ np.ones((1, p)))

    A = ZrStd1 @ ZrStd1.T
    A = np.clip(A, -1, 1)
    A = np.arccos(A)
    A[np.isnan(A)] = 0
    return np.sum(A) / (n * m**2)

def U_L_MT_ED(X, Y):
    m, p = X.shape
    n = Y.shape[0]
    IN1 = np.zeros(m)
    IN2 = np.zeros(m)
    IN3 = np.zeros(m)
    SA1 = np.zeros(n)
    SA2 = np.zeros(n)
    SA3 = np.zeros(n)

    for r in range(m):
        IN1[r] = Subtractr1(X, Y, r)
        IN2[r] = Subtractr3(X, r) / (m * (m - 1) * (m - 2))
        IN3[r] = Subtractr4(X, Y, r)
    for r in range(n):
        SA1[r] = Subtractr2(X, Y, r)
        SA2[r] = Subtractr3(Y, r) / (n * (n - 1) * (n - 2))
        SA3[r] = Subtractr5(X, Y, r)

    Indep_Index = (m * (np.sum(IN1) / 2 - np.sum(IN2)) + m * (np.sum(IN1) / 2 - np.sum(IN3)) +
                   n * (np.sum(SA1) / 2 - np.sum(SA2)) + n * (np.sum(SA1) / 2 - np.sum(SA3))) / (m + n)
    return Indep_Index

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
