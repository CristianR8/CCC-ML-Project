"""Class balancing helpers."""

from __future__ import annotations

from typing import Any


try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:  # pragma: no cover
    RandomOverSampler = None
    RandomUnderSampler = None
    SMOTE = None


def apply_sampling(
    X: Any,
    y: Any,
    method: str | None,
    random_state: int = 42,
):
    """Apply one of the supported sampling modes to training data only."""
    if method in (None, "none"):
        return X, y

    method = method.lower()
    if method == "over":
        if RandomOverSampler is None:
            raise ImportError("imblearn is required for RandomOverSampler")
        sampler = RandomOverSampler(random_state=random_state)
        return sampler.fit_resample(X, y)

    if method == "under":
        if RandomUnderSampler is None:
            raise ImportError("imblearn is required for RandomUnderSampler")
        sampler = RandomUnderSampler(random_state=random_state)
        return sampler.fit_resample(X, y)

    if method == "smote":
        if SMOTE is None:
            raise ImportError("imblearn is required for SMOTE")
        sampler = SMOTE(random_state=random_state, k_neighbors=5)
        return sampler.fit_resample(X, y)

    raise ValueError(f"Unsupported sampling method: {method}")
