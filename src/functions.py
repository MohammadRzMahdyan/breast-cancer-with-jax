import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import ConfusionMatrixDisplay

import jax.numpy as jnp
import jax.random as jr
import jax
from jax import grad, lax, tree_util
from jax.nn import sigmoid


def load_csv(path): 
    return pd.read_csv(path)

def split_xy(data, target): 
    return data.drop(columns=target), data[target]

def to_numpy(batch):
    X, y = batch
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy()
    return X, y

def fit_standard_scaler(x): 
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return mean, std


def transform_standard_scaler(state, x):
    mean, std = state
    return (x - mean) / np.where(std == 0, 1, std)


def fit_label_encoder(x): 
    return np.unique(x)

def transform_label_encoder(state, x):
    return np.searchsorted(state, x).astype(np.float32)

def split_train_test(batch):
    X, y = batch
    return tts(
        X,
        y,
        test_size=0.3,
        random_state=42,
        shuffle=True,
        stratify=y
    )

def pack_train_test(split_batch):
    X_train, X_test, y_train, y_test = split_batch
    return (
        (X_train, y_train),
        (X_test, y_test)
    )

def make_batches(X, y, batch_size):
    n = (len(X) // batch_size) * batch_size
    X = X[:n]
    y = y[:n]
    X = X.reshape(-1, batch_size, X.shape[1])
    y = y.reshape(-1, batch_size)
    return X, y

def init_params(n_features, seed=42):
    key = jr.PRNGKey(seed)
    return {
        "w": jr.normal(key, (n_features,)) * 0.01,
        "b": jnp.array(0.0)
    }

def linear(params, x): return jnp.dot(x, params["w"]) + params["b"]
    
def logits(params, x): return linear(params, x)

def predict(params, x): return sigmoid(logits(params, x))

def bce_with_logits(params, x, y):
    z = logits(params, x)
    return (
        jnp.maximum(z, 0)
        - z * y
        + jnp.log1p(jnp.exp(-jnp.abs(z)))
    )


def loss(params, x, y): return jnp.mean(bce_with_logits(params, x, y))

def grad_fn(params, x, y): return grad(loss)(params, x, y)

def update(params, grads, lr):
    return tree_util.tree_map(
        lambda p, g: p - lr * g,
        params,
        grads
    )

def step(carry, batch):
    params, lr = carry
    X, y = batch
    grads = grad_fn(params, X, y)
    params = update(params, grads, lr)
    l = loss(params, X, y)
    return (params, lr), l

def calculate_accuracy(
    params,
    X,
    y,
    threshold=0.5
):
    probs = predict(params, X)
    preds = (
        probs >= threshold
    ).astype(jnp.float32)
    return jnp.mean(preds == y)

def confusion_matrix(t, p):
    preds = (p >= 0.5).astype(float)
    return ConfusionMatrixDisplay.from_predictions(t, preds)