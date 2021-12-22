## Strategies

- v0
  - NN: YOLOv5s
    - CLI: `python yolov5/train.py --img 640 --batch 8 --epochs 100 --data gbr.yaml --weights yolov5s.pt --project gbr --name yolov5s`
    - TODO: add details -- loss and metrics plots.
  - Dataset: random split for train and val using full dataset.
    - TODO: add details about samples.
- v1
  - NN: YOLOv5s
  - Dataset: GroupKFold split based on sequence with limited background images -- WIP.
