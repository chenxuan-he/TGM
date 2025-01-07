import numpy as np
from sklearn.datasets import make_swiss_roll, make_blobs

def generate_standard_normal(n_samples, n_features=3, seed=0):
    """
    Generate data from a standard normal distribution.
    """
    np.random.seed(seed)
    return np.random.randn(n_samples, n_features)

def generate_uniform(n_samples, n_features=3, low=0, high=1, seed=0):
    """
    Generate data from a uniform distribution.
    """
    np.random.seed(seed)
    return np.random.uniform(low, high, size=(n_samples, n_features))

def generate_swiss_roll(n_samples, dim=3, noise=0.05, seed=0):
    """
    Generate data in the shape of a Swiss Roll.
    """
    np.random.seed(seed)
    data, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    if dim > 3:
        extra_dims = np.random.randn(n_samples, dim - 3)
        return np.hstack((data, extra_dims))
    return data

def generate_spherical_surface(n_samples, n_features=3, radius=1, seed=0):
    """
    Generate points uniformly on the surface of a sphere.
    """
    np.random.seed(seed)
    vec = np.random.randn(n_samples, n_features)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return radius * vec

def generate_helix(n_samples, dim=3, noise=0.1, seed=0):
    """
    Generate data in the shape of a 3D helix.
    """
    np.random.seed(seed)
    t = np.linspace(0, 4 * np.pi, n_samples)
    x = np.sin(t)
    y = np.cos(t)
    z = t
    data = np.stack((x, y, z), axis=1)
    if dim > 3:
        extra_dims = np.random.randn(n_samples, dim - 3) * noise
        return np.hstack((data, extra_dims))
    return data + np.random.normal(scale=noise, size=data.shape)

def generate_gaussian_mixture(n_samples, n_features=3, n_clusters=3, seed=0):
    """
    Generate data from a Gaussian mixture model.
    """
    np.random.seed(seed)
    data, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=seed)
    return data

def generate_torus(n_samples, dim=3, r_major=1, r_minor=0.3, seed=0):
    """
    Generate data in the shape of a torus.
    """
    np.random.seed(seed)
    theta = 2 * np.pi * np.random.rand(n_samples)
    phi = 2 * np.pi * np.random.rand(n_samples)
    x = (r_major + r_minor * np.cos(phi)) * np.cos(theta)
    y = (r_major + r_minor * np.cos(phi)) * np.sin(theta)
    z = r_minor * np.sin(phi)
    if dim > 3:
        base_data = np.stack((x, y, z), axis=1)
        extra_dims = np.random.randn(n_samples, dim - 3)
        return np.hstack((base_data, extra_dims))
    return np.stack((x, y, z), axis=1)

def generate_checkerboard(n_samples, n_features=3, noise=0.1, seed=0):
    """
    Generate data in a checkerboard pattern.
    """
    np.random.seed(seed)
    data = np.random.uniform(-2, 2, size=(n_samples, n_features))
    for i in range(1, n_features, 2):
        data[:, i] += (data[:, i - 1] > 0).astype(float)
    data += np.random.normal(scale=noise, size=data.shape)
    return data

def generate_low_dim_subspace(n_samples, n_features=3, subspace_dim=2, seed=0):
    """
    Generate data in a low-dimensional linear subspace embedded in a high-dimensional space.
    """
    np.random.seed(seed)
    subspace = np.random.randn(n_samples, subspace_dim)
    projection_matrix = np.random.randn(subspace_dim, n_features)
    return subspace @ projection_matrix

def generate_pinched_ellipse(n_samples, dim=2, noise=0.05, seed=0):
    """
    Generate data in the shape of a pinched ellipse.
    """
    np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    r = 1 + 0.5 * np.sin(2 * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    base_data = np.stack((x, y), axis=1)
    if dim > 2:
        extra_dims = np.random.randn(n_samples, dim - 2) * noise
        return np.hstack((base_data, extra_dims))
    return base_data + np.random.normal(scale=noise, size=base_data.shape)

def generate_spiral(n_samples, dim=3, noise=0.1, n_turns=3, seed=0):
    """
    Generate data in the shape of a 2D spiral.
    """
    np.random.seed(seed)
    t = np.linspace(0, 2 * np.pi * n_turns, n_samples)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t
    base_data = np.stack((x, y, z), axis=1)
    if dim > 3:
        extra_dims = np.random.randn(n_samples, dim - 3) * noise
        return np.hstack((base_data, extra_dims))
    return base_data + np.random.normal(scale=noise, size=base_data.shape)
