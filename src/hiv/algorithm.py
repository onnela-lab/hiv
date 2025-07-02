# Mostly copied from https://github.com/onnela-lab/summaries.

from __future__ import annotations
import numpy as np
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_X_y
from typing import Any, Optional


class NearestNeighborAlgorithm(BaseEstimator):
    """
    Draw approximate posterior samples using a nearest neighbor algorithm.

    Args:
        frac: Fraction of samples to return as approximate posterior samples (mutually
            exclusive with `n_samples).
        n_samples: Number of samples to draw (mutually exclusive with `frac`).
        minkowski_norm: Minkowski p-norm to use for queries (defaults to Euclidean
            distances).
        **kdtree_kwargs: Keyword arguments passed to the KDTree constructor.
    """

    def __init__(
        self,
        *,
        frac: float | None = None,
        n_samples: int | None = None,
        minkowski_norm: int = 2,
        **kdtree_kwargs,
    ) -> None:
        super().__init__()
        if (frac is None) == (n_samples is None):
            raise ValueError("Exactly one of `frac` and `n_samples` must be given.")
        self.frac = frac
        self.n_samples = n_samples
        self.minkowski_norm = minkowski_norm
        self.kdtree_kwargs = kdtree_kwargs

        self.tree_: Optional[KDTree] = None
        self.params_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray, params: np.ndarray) -> NearestNeighborAlgorithm:
        """
        Construct a :class:`.KDTree` for fast nearest neighbor search for sampling
        parameters.

        Args:
            data: Simulated data or summary statistics used to build the tree with shape
                (n_samples, n_summaries).
            params: Parameters used to generate the summary statistics with shape
                (n_samples, n_params).
        """
        data, params = check_X_y(data, params, multi_output=True)
        self.tree_ = KDTree(data, **self.kdtree_kwargs)
        self.params_ = params
        return self

    def predict(
        self, data: np.ndarray, return_features: bool = False, **kwargs: Any
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Draw approximate posterior samples.

        Args:
            data: Data to condition on with shape `(batch_size, n_features)`.
            return_features: Return the features matching parameters.
            **kwargs: Keyword arguments passed to the KDTree query method.

        Returns:
            Array of posterior samples with shape `(batch_size, n_samples, n_params)`.
            If `return_features` is truth-y, also return the corresponding features with
            shape `(batch_size, n_samples, n_features)` in a tuple.
        """
        # Validate the state and input arguments.
        if self.tree_ is None:
            raise NotFittedError

        data = check_array(data)
        n_samples = self.n_samples or int(self.frac * self.tree_.n)
        _, idx = self.tree_.query(data, k=n_samples, p=self.minkowski_norm, **kwargs)
        # Explicitly reshape because `query` drops one dimension if the number of
        # samples is one.
        idx = idx.reshape((*data.shape[:-1], n_samples))

        assert self.params_ is not None, "Nearest neighbor sampler has not been fitted."
        params = self.params_[idx]
        if return_features:
            return (params, self.tree_.data[idx])
        else:
            return params


def regression_adjust(
    regressor: RegressorMixin, X: np.ndarray, y: np.ndarray, X_obs: np.ndarray
) -> np.ndarray:
    """
    Apply regression adjustment using a scikit-learn regressor.

    Args:
        regressor: Scikit-learn model for regression adjustment.
        X: Summary statistics of candidate samples to be adjusted with shape (n_samples,
            n_summaries). The shape can also be (batch_size, n_samples, n_summaries) in
            which case the regression adjustment is explicitly broadcast along the first
            dimension.
        y: Candidate parameter samples to be adjusted with shape (n_samples,
            n_parameters). The same comment on batch shape above applies.
        X_obs: Observed summary statistics with shape (n_summaries,).  The same comment
            on batch shape above applies.

    Returns:
        Regression-adjusted parameters with shape (n_samples, n_parameters) or
        (batch_size, n_samples, n_parameters).
    """
    # Broadcast along the first dimension by iteration.
    if np.ndim(X) == 3:
        batch_size, _, _ = X.shape
        assert X_obs.shape[0] == batch_size
        assert y.shape[0] == batch_size
        return np.asarray(
            [regression_adjust(regressor, *args) for args in zip(X, y, X_obs)]
        )

    n_summary_samples, n_summaries = X.shape
    n_param_samples, _ = y.shape
    assert n_summary_samples == n_param_samples
    assert X_obs.shape == (n_summaries,)

    regressor.fit(X, y)  # type: ignore[reportAttributeAccessIssue]
    correction = regressor.predict(X_obs[None])[0] - regressor.predict(X)  # type: ignore[reportAttributeAccessIssue]
    return y + correction
