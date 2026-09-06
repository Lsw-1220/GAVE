import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from run.run_evaluate import run_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--param_file", required=True)
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--cache_dir", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    result_file = Path(args.result_file).resolve()
    result_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.param_file, "r", encoding="utf-8") as file:
        model_param = json.load(file)
    model_param["save_dir"] = str(model_dir)
    model_param["device"] = "cpu"
    model_param["test_cache_dir"] = str(Path(args.cache_dir).resolve())

    data_loader = TestDataLoader(
        file_path=args.test_csv,
        cache_dir=model_param["test_cache_dir"],
    )
    from run.checkpoint_selection import select_checkpoints
    model_param['test_csv'] = str(Path(args.test_csv).resolve())
    select_checkpoints(model_param, data_loader, result_file)


if __name__ == "__main__":
    main()
