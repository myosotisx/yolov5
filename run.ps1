# python .\AT_distill.py --weights yolov5l.pt --cfg models/yolov5c.yaml --data data/voc.yaml --workers 4 --batch-size 16 --epochs-distill 30 --epochs 0
# python .\train.py --weights runs/AT_distill/exp37/weights_distill/last_distill.pt --data data/voc.yaml --hyp data/hyp.distill.yaml --workers 4 --batch-size 32 --epochs 100
python .\train.py --resume runs/train/exp53/weights/last.pt --data data/voc.yaml --hyp data/hyp.distill.yaml --workers 4 --batch-size 32 --epochs 100

# python .\AT_train.py --cfg models/yolov5c.yaml --weights ' ' --data data/voc.yaml --batch-size 16 --workers 4 --epochs 100 --teacher yolov5l.pt
