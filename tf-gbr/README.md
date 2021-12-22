## Strategies

- v0
  - NN: YOLOv5s
    - CLI: `python yolov5/train.py --img 640 --batch 8 --epochs 100 --data gbr.yaml --weights yolov5s.pt --project gbr --name yolov5s`
    - TODO: add details -- loss and metrics plots.
  - Dataset: random split for train and val using full dataset.
    - TODO: add details about samples.
- v1
  - NN: YOLOv5s
    - CLI: `python yolov5/train.py --img 640 --batch 18 --epochs 100 --data gbr.yaml --weights yolov5s.pt --project gbr --name v1_yolov5s`
  - Dataset: GroupKFold split based on sequence.
- v2
  - NN: YOLOv5s
    - CLI: `python yolov5/train.py --img 640 --batch 18 --epochs 100 --data gbr.yaml --weights yolov5s.pt --project gbr --name v2_yolov5s`
  - Dataset: GroupKFold split based on sequence with limited (10%) background images per sequence.
    - Also changed n_splits to 80% (arbitrarily) of the number of unique sequences.
