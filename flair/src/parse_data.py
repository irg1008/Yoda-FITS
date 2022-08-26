import json
from os import path

import pandas as pd

from split_data import read_csv


def parse(data: pd.DataFrame) -> list[dict]:
    json_data = []

    cols = data.columns.to_list()

    for _, row in data.iterrows():
        json_data.append(
            {
                cols[0]: row[0],
                cols[1]: row[1],
            }
        )

    return json_data


def export(json_data: list[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(json_data, indent=4, sort_keys=True))


def main():
    data_folder = path.join(path.dirname(__file__), "../data/")
    splits_folder = data_folder + "splits/"
    out_folder = data_folder + "out/"

    for name in ["train", "val", "test"]:
        data = read_csv(splits_folder + name + ".csv")
        json_data = parse(data)
        export(json_data, out_folder + name + ".json")


if __name__ == "__main__":
    main()
