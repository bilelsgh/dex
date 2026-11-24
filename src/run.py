"""
Call the different DeX components without running the dashboard
"""
import argparse
import copy
import json
import sys
import time

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from helpers.data_preproccesing import run_preprocess, split_datasets
from helpers.dataset_tools import DatasetDownloader
from helpers.utils import SplitMethod, get_split_idx


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the DeX script.
     -C / --config: Path to the JSON configuration file (required)
     --train: Path to the training dataset (CSV) (required)
     --test: Path to the test dataset (CSV) (optional)
     Example usage:
     python run.py -C ./config.json --train ./data/train.csv --test ./data/test.csv
     python run.py -C ./config.json --train ./data/train.csv
    """
    parser = argparse.ArgumentParser(
        description="Script for loading JSON configuration and datasets",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-C",
        "--config",
        type=str,
        required=True,
        metavar="CHEMIN",
        help="Path to the JSON configuration file\nExample: -C ./config.json",
    )
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        metavar="CHEMIN",
        help="Path to the training dataset (CSV)\nExample: --train ./data/train.csv",
    )
    parser.add_argument(
        "--test",
        type=str,
        required=False,
        default=None,
        metavar="CHEMIN",
        help="Path to the test dataset (CSV), optional\nExample: --test ./data/test.csv",
    )

    parser.add_argument(
        "--split",
        type=int,
        required=False,
        default=0,
        metavar="METH",
        help="Split methodology:\n0: One merged dataset\n1: Same as uploaded\n2: Train / Test (80 / 20)\nExample: --split 2",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Run preprocessing 10 times and log the mean, standard deviation and "
        "95%% confidence interval of the duration instead of a single run.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """
    Load a JSON configuration file.
    :param path: path to JSON file
    :return: configuration as a dictionary
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: file not found → '{path}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Error: invalid JSON → {e}")
        sys.exit(1)


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    :param path: path to CSV file
    :return: DataFrame containing the dataset
    """
    try:
        df = pd.read_csv(path)
        logger.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"Error: file not found → '{path}'")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()

    # -- load parameters and datasets
    operations = load_config(args.config)
    logger.success("Configuration loaded:", operations)

    df_train = load_dataframe(args.train)
    dfs = [df_train]

    df_test = None
    if args.test is not None:
        df_test = load_dataframe(args.test)
        dfs.append(df_test)
    else:
        logger.warning("No test dataset provided.")

    # -- preprocess
    split_method = {i: m for i, m in enumerate(SplitMethod)}
    datasets_idx, datasets_name = get_split_idx(
        split_method[args.split],
        [args.train, args.test] if df_test is not None else [args.train],
        dfs,
    )
    logger.warning(datasets_name)

    merged_dataset = pd.concat(dfs, ignore_index=True)

    if args.perf:
        # Run preprocessing 10 times to get a mean/std/confidence interval of
        # its duration. `operations` is deep-copied each run since run_preprocess
        # mutates its "standardization" args in place.
        n_runs = 10
        durations = []
        for i in range(n_runs):
            run_start = time.perf_counter()
            preprocessed = run_preprocess(
                copy.deepcopy(operations), merged_dataset, datasets_idx
            )
            durations.append(time.perf_counter() - run_start)
            logger.debug(f"[perf] run {i + 1}/{n_runs}: {durations[-1]:.3f}s")

        durations = np.array(durations)
        mean = durations.mean()
        std = durations.std(ddof=1)
        ci_low, ci_high = stats.t.interval(
            0.95, df=n_runs - 1, loc=mean, scale=std / np.sqrt(n_runs)
        )
        logger.info(
            f"Preprocessing perf (operations={list(operations.keys())}, n={n_runs}): "
            f"mean={mean:.3f}s, std={std:.3f}s, 95% CI=[{ci_low:.3f}s, {ci_high:.3f}s]"
        )
    else:
        # Chronométrage du preprocessing, avec les opérations (config) utilisées.
        preprocess_start = time.perf_counter()

        preprocessed = run_preprocess(operations, merged_dataset, datasets_idx)

        preprocess_duration = time.perf_counter() - preprocess_start
        logger.info(
            f"Preprocessing (operations={list(operations.keys())}) took "
            f"{preprocess_duration:.2f}s."
        )

    logger.debug(
        f"Dataset preprocessed! About to prepare the download. Dataset indexes: {datasets_idx}"
    )
    res = DatasetDownloader(datasets_idx, preprocessed, datasets_name, True)
    buf = res.download_file()
