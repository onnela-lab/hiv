import argparse
import collectiontools
import numpy as np
import pathlib
import pickle
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from typing import cast, Literal, Sequence
from ..algorithm import NearestNeighborAlgorithm, regression_adjust
from ..util import Timer


class _Args:
    adjust: bool
    frac: float
    exclude: list[str]
    standardize: None | Literal["global", "local"]
    max_lag: int | None
    save_samples: bool
    train: pathlib.Path
    test: pathlib.Path
    output: pathlib.Path


def load_batches(
    path: pathlib.Path,
    exclude_summaries: Sequence[str] | None = None,
    exclude_params: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load batches of summaries and parameters from a directory.

    Args:
        path: Directory containing batches as *.pkl files.
        exclude_summaries: Summary statistics to exclude.
        exclude_params: Parameters to exclude.
    """
    assert path.is_dir()
    exclude_summaries = exclude_summaries or ()
    exclude_params = exclude_params or ()

    summaries = {}
    params = {}

    for filename in path.glob("*.pkl"):
        with filename.open("rb") as fp:
            result = pickle.load(fp)

        summaries_batch = {
            key: value
            for key, value in result["summaries"].items()
            if not (key.startswith("_") or key in exclude_summaries)
        }
        collectiontools.append_values(summaries, summaries_batch)

        params_batch = {
            key: value
            for key, value in result["params"].items()
            if not (key == "n" or key in exclude_params)
        }
        collectiontools.append_values(params, params_batch)

    summaries = collectiontools.map_values(np.concatenate, summaries)
    params = collectiontools.map_values(np.concatenate, params)
    return summaries, params


def flatten_dict(X: dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray:
    # Turn into tensor and move the dict items to the last dimension.
    return np.moveaxis(np.asarray([X[key] for key in keys]), 0, -1)


def __main__(argv: list[str] | None = None) -> None:
    """
    Run approximate Bayesian computation.
    """
    if argv:
        argv = list(map(str, argv))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adjust", "-a", help="Apply regression adjustment.", action="store_true"
    )
    parser.add_argument(
        "--frac",
        "-f",
        type=float,
        help="Fraction of reference table to sample.",
        default=0.01,
    )
    parser.add_argument(
        "--exclude",
        "-e",
        help="Exclude summary statistic from ABC.",
        action="append",
    )
    parser.add_argument("--max-lag", "-m", help="Maximum lag to consider.", type=int)
    parser.add_argument(
        "--standardize", "-s", help="Standardize features.", choices={"local", "global"}
    )
    parser.add_argument(
        "--save-samples",
        help="Save samples in result, otherwise only diagnostics.",
        action="store_true",
    )
    parser.add_argument(
        "train",
        help="Directory containing batches of the reference table as *.pkl files.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "test",
        help="Directory containing batches of test data as *.pkl files.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output", help="File to save posterior samples.", type=pathlib.Path
    )
    args = cast(_Args, parser.parse_args(argv))

    if args.exclude:
        print(f"Excluded features: {', '.join(args.exclude)}")

    # Load and flatten features and parameters.
    timer = Timer()
    with timer("train"):
        train_features, train_params = load_batches(args.train, args.exclude)
        feature_names = tuple(train_features)
        param_names = tuple(train_params)
        n_params = len(param_names)

        train_features = flatten_dict(train_features, feature_names)
        train_params = flatten_dict(train_params, param_names)

        assert train_features.ndim == 3
        n_train_samples, n_train_lags, n_features = train_features.shape

    print(
        f"Loaded {n_train_samples:,} training samples with {n_train_lags} lags, "
        f"{n_features} features ({', '.join(feature_names)}), and {n_params} "
        f"parameters ({', '.join(param_names)}) from "
        f"{len(list(args.train.glob('*.pkl')))} files in '{args.train}', taking "
        f"{timer.times['train']:.1f} seconds."
    )

    with timer("test"):
        test_features, test_params = load_batches(args.test, args.exclude)
        assert feature_names == tuple(test_features)
        assert param_names == tuple(test_params)

        test_features = flatten_dict(test_features, feature_names)
        test_params = flatten_dict(test_params, param_names)

        n_test_samples, n_test_lags, _ = test_features.shape
        assert n_test_lags == n_train_lags

    print(
        f"Loaded {n_test_samples:,} test samples from "
        f"{len(list(args.test.glob('*.pkl')))} files in '{args.test}', taking "
        f"{timer.times['test']:.1f} seconds."
    )

    # Apply feature standardization if desired.
    if args.standardize is None:
        loc = 0
        scale = 1
    elif args.standardize == "global":
        loc = train_features.mean(axis=(0, 1))
        scale = train_features.std(axis=(0, 1))
    elif args.standardize == "local":
        loc = train_features.mean(axis=0)
        scale = train_features.std(axis=0)
    else:
        raise ValueError(args.standardize)  # pragma: no cover

    # Fix division by zero errors for constant features.
    scale = np.where(scale > 1e-12, scale, 1)
    train_features = (train_features - loc) / scale
    test_features = (test_features - loc) / scale

    # Sample the posterior for each lag.
    n_lags = n_train_lags
    if args.max_lag:
        n_lags = min(n_lags, args.max_lag)

    n_posterior_samples = int(args.frac * n_train_samples)

    # Sanity check the memory requirements.
    expected_gb = 8 * n_lags * n_posterior_samples * n_test_samples * n_params / 1e9
    if args.save_samples and expected_gb > 4:  # pragma: no cover
        print(f"WARNING: Saving samples will consume {expected_gb:.1f} GB.")
        response = input("Continue? (y/*n*)")
        if not response.startswith("y"):
            print("Exiting ...")
            return

    samples = []
    mses = []
    for lag in tqdm(range(n_lags)):
        # Draw samples using rejection ABC.
        sampler = NearestNeighborAlgorithm(frac=args.frac)
        sampler.fit(train_features[:, lag], train_params)
        param_samples, feature_samples = sampler.predict(
            test_features[:, lag], return_features=True
        )

        # Apply linear regression adjustment if requested. We apply it in the logit
        # space so we're on the full real line.
        if args.adjust:
            logit_samples = logit(param_samples)
            logit_samples = regression_adjust(
                LinearRegression(),
                feature_samples,
                logit_samples,
                test_features[:, lag],
            )
            param_samples = expit(logit_samples)

        # Evaluate the MSE for this lag. We do the evaluation here because
        assert param_samples.shape == (n_test_samples, n_posterior_samples, n_params)
        mses.append(np.square(param_samples - test_params[:, None, :]).mean(axis=1))

        if args.save_samples:
            samples.append(param_samples)

    if args.save_samples:
        # Starts out with (n_lags, n_test_samples, n_posterior_samples, n_params). We
        # want (..., n_test_samples, n_params) so we can easily do the evaluation of
        # diagnostics. A sensible format might be
        # (n_lags, n_posterior_samples, n_test_samples, n_params). But there is some
        # ambiguity on whether the posterior samples or lags should lead.
        samples = np.swapaxes(np.stack(samples), 1, 2)
        assert samples.shape == (n_lags, n_posterior_samples, n_test_samples, n_params)

    mses = np.stack(mses)
    assert mses.shape == (n_lags, n_test_samples, n_params)

    # Save the results, including the simulated parameter values and posterior samples
    # if desired.
    result = {
        "args": vars(args),
        "param_names": param_names,
        "mses": mses,
        "standardize": {"loc": loc, "scale": scale},
    }
    if args.save_samples:
        result.update(
            {
                "params": test_params,
                "samples": samples,
            }
        )
    with open(args.output, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":
    __main__()
