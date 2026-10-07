"""Embedding-space generation metrics and height-normalized positional error."""

from __future__ import annotations

import numpy as np


def cosine_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    assert (first_norm > 0).all() and (second_norm > 0).all()
    return (first / first_norm) @ (second / second_norm).T


def retrieval(text: np.ndarray, motion: np.ndarray, pool_size: int = 100) -> dict[str, float]:
    assert pool_size >= 3 and len(text) == len(motion)
    count = len(text) // pool_size * pool_size
    assert count > 0, "Not enough motions for a complete retrieval pool."
    correct, score = np.zeros(3), 0.0
    for start in range(0, count, pool_size):
        similarities = cosine_matrix(text[start:start + pool_size], motion[start:start + pool_size])
        ranks = np.argsort(-similarities, axis=1)
        matches = ranks == np.arange(pool_size)[:, None]
        correct += np.array([matches[:, :k].any(1).sum() for k in (1, 2, 3)])
        score += np.trace(similarities)
    return {
        "r_precision_1": float(correct[0] / count),
        "r_precision_2": float(correct[1] / count),
        "r_precision_3": float(correct[2] / count),
        "clip_score": float(score / count),
    }


def fid(real: np.ndarray, generated: np.ndarray) -> float:
    """Symmetric PSD formulation of Gaussian Fréchet distance, in float64."""
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    assert real.ndim == generated.ndim == 2 and len(real) > 1 and len(generated) > 1
    mean1, mean2 = real.mean(0), generated.mean(0)
    covariance1, covariance2 = np.cov(real, rowvar=False), np.cov(generated, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance1)
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    middle = root @ covariance2 @ root
    trace = np.sqrt(np.maximum(np.linalg.eigvalsh((middle + middle.T) / 2), 0)).sum()
    value = (
        ((mean1 - mean2) ** 2).sum()
        + np.trace(covariance1) + np.trace(covariance2) - 2 * trace
    )
    return float(max(value, 0))


def diversity(values: np.ndarray, rng: np.random.Generator, pairs: int = 300) -> float:
    assert len(values) > 1
    pairs = min(pairs, len(values) - 1)
    first, second = (rng.choice(len(values), pairs, replace=False) for _ in range(2))
    return float(np.linalg.norm(values[first] - values[second], axis=-1).mean())


def multimodality(values: np.ndarray, rng: np.random.Generator, pairs: int = 10) -> float:
    assert values.ndim == 3 and values.shape[1] > pairs
    first, second = (rng.choice(values.shape[1], pairs, replace=False) for _ in range(2))
    return float(np.linalg.norm(values[:, first] - values[:, second], axis=-1).mean())


def position_error(prediction: np.ndarray, target: np.ndarray, height: float) -> float:
    assert prediction.shape == target.shape and height > 0
    return float(np.square(prediction - target).sum(-1).mean() / height ** 2 * 1000)


def summarize(replications: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    assert replications
    result = {}
    for metric in replications[0]:
        values = np.array([replication[metric] for replication in replications])
        ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        result[metric] = {"mean": float(values.mean()), "ci95": float(ci)}
    return result
