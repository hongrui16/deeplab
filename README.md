# pytorch-deeplab (semantic segmentation, distributed training)


| Backbone  | train/eval os  |mIoU in val |Pretrained Model|
| :-------- | :------------: |:---------: |:--------------:|
| ResNet    | 16/16          | 78.43%     | [google drive](https://drive.google.com/open?id=1NwcwlWqA-0HqAPk3dSNNPipGMF0iS0Zu) |
| MobileNet | 16/16          | 70.81%     | [google drive](https://drive.google.com/open?id=1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt) |
| DRN       | 16/16          | 78.87%     | [google drive](https://drive.google.com/open?id=131gZN_dKEXO79NknIQazPJ-4UmRrZAfI) |

## Introduction
This is a PyTorch implementation of [DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611). It
can use Modified Aligned Xception and ResNet as backbone. Currently, we train DeepLab V3 Plus
using Pascal VOC 2012, SBD, Cityscapes, and customer-made datasets.

## How to train
### 1 Use slurm
#### 1.1 foreground running
srun -N1 --cpus-per-task 32 --gres gpu:4 python train_main.py
#### 1.2 background running
sbatch background_running.slurm
### 2 directly use gpu machines
CUDA_VISIBLE_DEVICES=0.1.2.3 python train_main.py

## 如何提交代码
1.git init

#初始化本地仓库

2.git remote add origin https://github.com/hongrui16/deeplab.git(或则 git:git的地址)

#关联本地仓库到远程仓库

git add *
#添加要提交的文件到暂存区

4.git commit -m "init commint"

#提交代码到文件控制仓库

5.git fetch origin

#将远程主机的更新，全部取回本地

6.git pull origin main 如果报错用这个 git pull origin main --allow-unrelated-histories

#拉取远程分支代码到本地

7.git push -u origin main

#提交本地分支(main)代码到远程分支(main)

本地仓库的东西push到远程仓库时报错：

看网上很多人说是因为本地仓库的文件和远程仓库不一样，所以要先用命令git pull -f origin main将远程仓库的文件拉到本地：

可是这样做之后再git push还是没有用，还是报同样的错误。 于是就用git push -f origin main强制push就成功了。（注意：大家千万不要随便用-f的操作，因为f意味着强制push，会覆盖掉远程的所有代码！）

## Acknowledgement
[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception.git)

[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

[drn](https://github.com/fyu/drn)
