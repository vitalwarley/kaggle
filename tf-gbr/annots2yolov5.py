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


def gen_annotations(df, split):
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_path = (
            Path("train_images") / f"video_{row.video_id}" / f"{row.video_frame}.jpg"
        )
        annotations = json.loads(row.annotations.replace("'", '"'))

        # link image, write labels
        sample_name = f"{row.video_id}_{row.video_frame}"
        # TODO: force symlink if it exists; otherwise checks existences, then delete it, then create it again.
        Path(f"dataset/{args.version}/images/{split}/{sample_name}.jpg").symlink_to(
            image_path.absolute()
        )
        with open(f"dataset/{args.version}/labels/{split}/{sample_name}.txt", "w") as f:
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()

    full = pd.read_csv("train.csv")
    train, other, _, _ = train_test_split(full, full.sequence, test_size=0.4, random_state=42)
    val, test, _, _ = train_test_split(other, other.sequence, test_size=0.5, random_state=42)

    for split in ["train" , "val" , "test"]:
        Path(f"dataset/{args.version}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"dataset/{args.version}/{split}").mkdir(parents=True, exist_ok=True)

    gen_annotations(train, "train", args.version)
    gen_annotations(val, "val", args.version)
    gen_annotations(test, "test", args.version)
