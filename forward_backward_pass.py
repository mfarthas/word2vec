import numpy as np
from model import sigmoid


def forward(
    center_idx: int,
    context_idx: int,
    neg_indices: np.ndarray,
    W_in: np.ndarray,
    W_out: np.ndarray
):
    v_c = W_in[center_idx]
    u_o = W_out[context_idx]
    u_neg = W_out[neg_indices]
    sig_pos = sigmoid(u_o @ v_c)
    sig_neg = sigmoid(u_neg @ v_c)
    return v_c, u_o, u_neg, sig_pos, sig_neg


def compute_loss(
    sig_pos: float,
    sig_neg: np.ndarray
) -> float:
    eps = 1e-10
    loss = (
        -np.log(sig_pos + eps)
        - np.sum(np.log(1 - sig_neg + eps))
    )
    return loss


def backward(
    v_c: np.ndarray,
    u_o: np.ndarray,
    u_neg: np.ndarray,
    sig_pos: float,
    sig_neg: np.ndarray
):
    grad_v_c = (
        -(1 - sig_pos) * u_o
        + (sig_neg @ u_neg)
    )
    grad_u_o = -(1 - sig_pos) * v_c
    grad_u_neg = sig_neg[:, np.newaxis] * v_c[np.newaxis, :]
    return grad_v_c, grad_u_o, grad_u_neg


def sgd_update(
    center_idx: int,
    context_idx: int,
    neg_indices: np.ndarray,
    grad_v_c: np.ndarray,
    grad_u_o: np.ndarray,
    grad_u_neg: np.ndarray,
    W_in: np.ndarray,
    W_out: np.ndarray,
    lr: float
):
    W_in[center_idx] -= lr * grad_v_c
    W_out[context_idx] -= lr * grad_u_o
    for i, neg_idx in enumerate(neg_indices):
        W_out[neg_idx] -= lr * grad_u_neg[i]


def step(
    center_idx: int,
    context_idx: int,
    neg_indices: np.ndarray,
    W_in: np.ndarray,
    W_out: np.ndarray,
    lr: float
) -> float:
    v_c, u_o, u_neg, sig_pos, sig_neg = forward(
        center_idx, context_idx, neg_indices, W_in, W_out
    )
    loss = compute_loss(sig_pos, sig_neg)
    grad_v_c, grad_u_o, grad_u_neg = backward(
        v_c, u_o, u_neg, sig_pos, sig_neg
    )
    sgd_update(
        center_idx, context_idx, neg_indices,
        grad_v_c, grad_u_o, grad_u_neg,
        W_in, W_out, lr
    )
    return loss
