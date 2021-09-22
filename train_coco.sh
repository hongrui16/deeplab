python train.py --backbone mobilenet --lr 0.01 --workers 4 --epochs 40 --batch-size 128 --gpu-ids 0,1  --eval-interval 1 --dataset coco
# python train.py --backbone mobilenet --lr 0.01 --workers 4 --epochs 40 --batch-size 2 --checkname deeplab-resnet --eval-interval 1 --dataset coco --no-cuda

