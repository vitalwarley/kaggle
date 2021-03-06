import sys
import json
from pathlib import Path
from argparse import ArgumentParser

import imagesize
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GroupKFold

sys.path.insert(0, str(Path(".").absolute() / "yolov5"))
from yolov5.utils.general import xyxy2xywhn, xywh2xyxy


def gen_annotations(df, split, version):
    if df is None:
        print(f"No dataframe was passed for split={split}.")
        return

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_path = (
            Path("train_images") / f"video_{row.video_id}" / f"{row.video_frame}.jpg"
        )
        annotations = json.loads(row.annotations.replace("'", '"'))

        # link image, write labels
        sample_name = f"{row.video_id}_{row.video_frame}"
        # TODO: force symlink if it exists; otherwise checks existences, then delete it, then create it again.
        link = Path(f"dataset/{version}/images/{split}/{sample_name}.jpg")
        if not link.exists():
            link.symlink_to(image_path.absolute())
        with open(f"dataset/{version}/labels/{split}/{sample_name}.txt", "w") as f:
            for annot in annotations:
                width, height = imagesize.get(image_path)
                cls = 0
                bbox = [list(annot.values())]
                bbox = np.array(bbox, np.float32)
                bbox[:, 2] += bbox[:, 0]
                bbox[:, 3] += bbox[:, 1]
                bbox = xyxy2xywhn(bbox, w=width, h=height)
                cx, cy, w, h = bbox[0]
                f.write(f"0 {cx} {cy} {w} {h}\n")


def create_yaml(version):
    contents = (
        f"path: dataset/{version}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test:  images/test\n"
        "nc: 1  # number of classes\n"
        "names: ['starfish']\n"
    )
    with open("gbr.yaml", "w") as f:
        f.write(contents)


if __name__ == "__main__":
    SEED = 42

    parser = ArgumentParser()
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--renew-dataset", action='store_true')
    args = parser.parse_args()

    create_yaml(args.version)

    if args.renew_dataset:
        import shutil
        shutil.rmtree(f"dataset/{args.version}", ignore_errors=True)

    for split in ["train", "val", "test"]:
        Path(f"dataset/{args.version}/images/{split}").mkdir(
            parents=True, exist_ok=True
        )
        Path(f"dataset/{args.version}/labels/{split}").mkdir(
            parents=True, exist_ok=True
        )

    full = pd.read_csv("train.csv")
    train, val, test = None, None, None
    if args.version == "v0":
        train, other, _, _ = train_test_split(
            full, full.sequence, test_size=0.4, random_state=SEED
        )
        val, test, _, _ = train_test_split(
            other, other.sequence, test_size=0.5, random_state=SEED
        )
    elif args.version == "v1":
        gkf = GroupKFold(n_splits=2)  # train, val
        train_idx, val_idx = next(
            gkf.split(full, full.sequence, groups=full.sequence)
        )  # skips 2nd split
        train = full[full.index.isin(train_idx)]
        val = full[full.index.isin(val_idx)]

        train.to_csv(f"dataset/{args.version}/train.csv", index=False)
        val.to_csv(f"dataset/{args.version}/val.csv", index=False)
    elif args.version == "v2":
        full["has_annots"] = full.annotations != "[]"
        # I want all has_annots = True, but only 10% of has_annots=False
        full = pd.concat(
            [
                full[full.has_annots == True],
                full[full.has_annots == False].groupby("sequence").sample(frac=0.1),
            ],
        ).reset_index()
        n_seqs = full.sequence.unique().shape[0]
        gkf = GroupKFold(n_splits=int(0.8 * n_seqs))  # train, val
        train_idx, val_idx = next(
            gkf.split(full, full.sequence, groups=full.sequence)
        )  # skips 2nd split
        train = full[full.index.isin(train_idx)]
        val = full[full.index.isin(val_idx)]

        train.to_csv(f"dataset/{args.version}/train.csv", index=False)
        val.to_csv(f"dataset/{args.version}/val.csv", index=False)
    else:
        raise Exception("Dataset version not implemented.")

    gen_annotations(train, "train", args.version)
    gen_annotations(val, "val", args.version)
    gen_annotations(test, "test", args.version)
