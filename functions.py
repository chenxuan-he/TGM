import numpy as np
import torch
from torch import nn

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


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=128, output_dim=1, binary=False, positive=False, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num//2, bias=True)
        self.fc3 = nn.Linear(hidden_num//2, output_dim, bias=True)
        if activation=="relu":
            self.act = lambda x: torch.relu(x)
        elif activation=="sigmoid":
            self.act = lambda x: torch.sigmoid(x)
        self.binary = binary
        self.positive = positive
    
    def forward(self, x_input):
        inputs = x_input
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        if self.binary:
            x = torch.sigmoid(x)
        elif self.positive:
            x = torch.relu(x)
        return x
    
class RectifiedFlow():
    def __init__(self, model=None, num_steps=1000):
        self.model = model
        self.N = num_steps
    
    def get_train_tuples(self, z0=None, z1=None):
        t = torch.rand((z1.shape[0], 1))
        z_t =  t * z1 + (1.-t) * z0
        target = z1 - z0 
        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, z0=None, N=None):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N    
        dt = 1./N
        traj = [] # to store the trajectory
        z = z0.detach().clone()
        batchsize = z.shape[0]
        
        traj.append(z.detach().clone())
        for i in range(N):
            t = torch.ones((batchsize, 1)) * i / N
            pred = self.model(torch.cat([z, t], dim=1))
            z = z.detach().clone() + pred * dt
        
            traj.append(z.detach().clone())

        return traj
    
def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
    loss_curve = []
    for i in range(inner_iters+1):
        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        z_t, t, target = rectified_flow.get_train_tuples(z0=z0, z1=z1)

        pred = rectified_flow.model(torch.cat([z_t, t], dim=1))
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()
        
        optimizer.step()
        loss_curve.append(np.log(loss.item())) ## to store the loss curve

    return rectified_flow, loss_curve