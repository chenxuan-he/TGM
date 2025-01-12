import numpy as np

def subtract_r1(X, Y, r):
    m, p = X.shape
    n = Y.shape[0]
    Zctr1 = Y - np.ones((n, 1)) @ X[r, :].reshape(1, p)
    ZrNorm1 = np.sqrt(np.sum(Zctr1**2, axis=1)).reshape(n, 1)
    ZrStd1 = Zctr1 / (ZrNorm1 @ np.ones((1, p)))

    Zctr2 = X - np.ones((m, 1)) @ X[r, :].reshape(1, p)
    ZrNorm2 = np.sqrt(np.sum(Zctr2**2, axis=1)).reshape(m, 1)
    ZrStd2 = Zctr2 / (ZrNorm2 @ np.ones((1, p)))

    A = ZrStd1 @ ZrStd2.T
    A = np.clip(A, -1, 1)
    A = np.arccos(A)
    A[np.isnan(A)] = 0
    return 2 * np.sum(A) / (m * (m - 1) * n)

def subtract_r2(X, Y, r):
    m, p = X.shape
    n = Y.shape[0]
    Zctr1 = X - np.ones((m, 1)) @ Y[r, :].reshape(1, p)
    ZrNorm1 = np.sqrt(np.sum(Zctr1**2, axis=1)).reshape(m, 1)
    ZrStd1 = Zctr1 / (ZrNorm1 @ np.ones((1, p)))

    Zctr2 = Y - np.ones((n, 1)) @ Y[r, :].reshape(1, p)
    ZrNorm2 = np.sqrt(np.sum(Zctr2**2, axis=1)).reshape(n, 1)
    ZrStd2 = Zctr2 / (ZrNorm2 @ np.ones((1, p)))

    A = ZrStd1 @ ZrStd2.T
    A = np.clip(A, -1, 1)
    A = np.arccos(A)
    A[np.isnan(A)] = 0
    return 2 * np.sum(A) / (n * (n - 1) * m)

def subtract_r3(Z, r):
    n, p = Z.shape
    Zctr = Z - np.ones((n, 1)) @ Z[r, :].reshape(1, p)
    Zctr = np.delete(Zctr, r, axis=0)

    ZrNorm = np.sqrt(np.sum(Zctr**2, axis=1)).reshape(n - 1, 1)
    ZrStd = Zctr / (ZrNorm @ np.ones((1, p)))
    A = ZrStd @ ZrStd.T
    A = np.clip(A, -1, 1)
    A = np.arccos(A)
    A[np.isnan(A)] = 0
    return np.sum(A) - np.sum(np.diag(A))

def subtract_r4(X, Y, r):
    m, p = X.shape
    n = Y.shape[0]
    Zctr1 = Y - np.ones((n, 1)) @ X[r, :].reshape(1, p)
    ZrNorm1 = np.sqrt(np.sum(Zctr1**2, axis=1)).reshape(n, 1)
    ZrStd1 = Zctr1 / (ZrNorm1 @ np.ones((1, p)))

    A = ZrStd1 @ ZrStd1.T
    A = np.clip(A, -1, 1)
    A = np.arccos(A)
    A[np.isnan(A)] = 0
    return np.sum(A) / (m * n**2)

def subtract_r5(X, Y, r):
    m, p = X.shape
    n = Y.shape[0]
    Zctr1 = X - np.ones((m, 1)) @ Y[r, :].reshape(1, p)
    ZrNorm1 = np.sqrt(np.sum(Zctr1**2, axis=1)).reshape(m, 1)
    ZrStd1 = Zctr1 / (ZrNorm1 @ np.ones((1, p)))

    A = ZrStd1 @ ZrStd1.T
    A = np.clip(A, -1, 1)
    A = np.arccos(A)
    A[np.isnan(A)] = 0
    return np.sum(A) / (n * m**2)

def u_l_mt_ed(X, Y):
    m, p = X.shape
    n = Y.shape[0]
    IN1 = np.zeros(m)
    IN2 = np.zeros(m)
    IN3 = np.zeros(m)
    SA1 = np.zeros(n)
    SA2 = np.zeros(n)
    SA3 = np.zeros(n)

    for r in range(m):
        IN1[r] = subtract_r1(X, Y, r)
        IN2[r] = subtract_r3(X, r) / (m * (m - 1) * (m - 2))
        IN3[r] = subtract_r4(X, Y, r)

    for r in range(n):
        SA1[r] = subtract_r2(X, Y, r)
        SA2[r] = subtract_r3(Y, r) / (n * (n - 1) * (n - 2))
        SA3[r] = subtract_r5(X, Y, r)

    Indep_Index = (m * (np.sum(IN1) / 2 - np.sum(IN2)) + m * (np.sum(IN1) / 2 - np.sum(IN3)) +
                   n * (np.sum(SA1) / 2 - np.sum(SA2)) + n * (np.sum(SA1) / 2 - np.sum(SA3))) / (m + n)
    return Indep_Index