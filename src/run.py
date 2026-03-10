"""
Call the different DeX components without running the dashboard
"""
import argparse
import json
import sys

import pandas as pd
from loguru import logger

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
    preprocessed = run_preprocess(
        operations, pd.concat(dfs, ignore_index=True), datasets_idx
    )

    res = DatasetDownloader(datasets_idx, preprocessed, datasets_name, True)
    buf = res.download_file
