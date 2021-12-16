# pytorch-deeplab (semantic segmentation, distributed training)


## Introduction
This is a PyTorch implementation of [DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611). It
can use Modified Aligned Xception and ResNet as backbone. Currently, we train DeepLab V3 Plus
using Pascal VOC 2012, SBD, Cityscapes, and basicDataset(customer-made) datasets.


## Prepare basicDataset(customer-made) datasets

```
/datasets
    /train
        /image
            a.jpg
            ...
        /label
            a.png
            ...
    /val
        /image
            b.jpg
            ...
        /label
            b.png
            ...
    /test
        /image
            c.jpg
            ...
        /label
            c.png
            ...
```
## Metro Label Name to Value
```
metro_label_name_to_value = {"left_1": 1, "right_1": 2, "left_2": 3, "right_2": 4, 
                                         "left_3": 5, "right_3": 6, "left_4": 7, "right_4": 8,
                                         "left_5": 9, "right_5": 10, "left_6": 11, "right_6": 12}
for shape in sorted(data["shapes"], key=lambda x: x["label"]):
    label_name = shape["label"]
    if label_name in metro_label_name_to_value:
        label_value = (metro_label_name_to_value[label_name])*20
        metro_label_name_to_value[label_name] = label_value
    else:
        label_value = 250
        metro_label_name_to_value[label_name] = label_value
```


## Train, Val, and Test selection
args.testValTrain: '-1: no, 0: infer, 1: test, 2: train, 3: trainval, 4: trainvaltest'
```
no: do nothing
infer: only do inference
test: do inference and calculate metrics such as miou and fwiou
...
```

## How to train
### 1 Use slurm
#### 1.1 foreground running
```
srun -N1 --cpus-per-task 16 --gres gpu:4 python train_main.py --testValTrain 4
```
#### 1.2 background running
set "args.testValTrain = 2, 3, or 4" and then run the command below
```
sbatch background_running.slurm
```
### 2 directly use gpu machines
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_main.py --testValTrain 4 --resume run/basicDataset/deeplab-resnet/experiment_7/checkpoint.pth.tar --loss_type ce
```
## How to test or val
### 1 Use slurm
#### 1.1 foreground running
```
srun -N1 --cpus-per-task 8 --gres gpu:4 python train_main.py --testValTrain 1 --resume run/basicDataset/deeplab-resnet/experiment_*/model_best.pth.tar 
```
#### 1.2 background running
set "args.testValTrain = 1" and then run the command below
```
sbatch background_running.slurm
```
### 2 directly use gpu machines
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_main.py --testValTrain 1 --resume run/basicDataset/deeplab-resnet/experiment_*/model_best.pth.tar --dump_image
```
## Inference only
This is for development.
```
python inference.py --resume run/basicDataset/deeplab-resnet/experiment_*/model_best.pth.tar --gpu_id .......
```
## Acknowledgement
[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception.git)

[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

[drn](https://github.com/fyu/drn)


## How to push and pull code on GitHub
1. 初始化本地仓库
```
git init
```
2. 关联本地仓库到远程仓库
```
git remote add origin https://github.com/User/Repo.git
```
3. 添加要提交的文件到暂存区
```
git add *
```
4. 提交代码到文件控制仓库
```
git commit -m "init commint"
```
5. 将远程主机的更新，全部取回本地
```
git fetch origin
```

6. 拉取远程分支代码到本地
```
git pull origin main 
```
如果报错用这个 
```
git pull origin main --allow-unrelated-histories
```

7. 提交本地分支(main)代码到远程分支(main)
```
git push -u origin main
```

### Q: 本地仓库的东西push到远程仓库时报错

A: 因为本地仓库的文件和远程仓库不一样，所以要先用命令git pull -f origin main将远程仓库的文件拉到本地. 可是这样做之后再git push还是没有用，还是报同样的错误。 于是就用git push -f origin main强制push就成功了。（注意：大家千万不要随便用-f的操作，因为f意味着强制push，会覆盖掉远程的所有代码！）
