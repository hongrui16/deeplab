# CUDA_VISIBLE_DEVICES=5,6,7 python train_main.py --testValTrain 4 --use_txtfile
# CUDA_VISIBLE_DEVICES=5,6,7 python train_main.py --testValTrain 4 --use_txtfile --base_size 720 --crop_size 720
# CUDA_VISIBLE_DEVICES=2,3 python train_main.py --testValTrain 4 --globally_distinguish_left_right
# srun -N1 --cpus-per-task 4 --gres gpu:4 python train_main.py --testValTrain 4 --globally_distinguish_left_right
# CUDA_VISIBLE_DEVICES=2,3 python train_main.py --testValTrain 4 --distinguish_left_right_semantic


###pot project
# CUDA_VISIBLE_DEVICES=2 python train_main.py --testValTrain 4 --use_txtfile --base_size 200 --crop_size 200 --dataset CustomPot --n_classes 2 \
# --dataset_dir /home/hongrui/project/metro_pro/dataset/pot --rotate_degree 30 --epochs 500 --pot_train_mode 1 \
# --resume run/CustomPot/deeplab-resnet/experiment_0/model_best.pth.tar 

# CUDA_VISIBLE_DEVICES=3 python train_main.py --testValTrain 4 --use_txtfile --base_size 200 --crop_size 200 --dataset CustomPot --n_classes 2 \
# --dataset_dir /home/hongrui/project/metro_pro/dataset/pot --rotate_degree 30 --epochs 500 \
# --pot_train_mode 4

# CUDA_VISIBLE_DEVICES=1 python train_main.py --testValTrain 4 --use_txtfile --base_size 640 --crop_size 640 --dataset CustomPot --n_classes 4 \
# --dataset_dir /home/hongrui/project/metro_pro/dataset/pot --rotate_degree 30 --epochs 500 \
# --diff_all_classes --ignore_huahen

# CUDA_VISIBLE_DEVICES=2 python train_main.py --testValTrain 4 --use_txtfile --base_size 640 --crop_size 640 --dataset CustomPot --n_classes 3 \
# --dataset_dir /home/hongrui/project/metro_pro/dataset/pot --rotate_degree 30 --epochs 500 \
# --diff_all_classes --ignore_huahen --ignore_zhoubian


###GC10-DET project
# CUDA_VISIBLE_DEVICES=1 python train_main.py --testValTrain 4  --base_size 420 --crop_size 420  --batch_size 16 --dataset GC10_DET  \
# --dataset_dir /comp_robot/hongrui/pot_pro/GC10-DET --rotate_degree 30 --epochs 300 \
# --pot_train_mode 3

# CUDA_VISIBLE_DEVICES=2 python train_main.py --testValTrain 4  --base_size 640 --crop_size 640 --dataset GC10_DET --n_classes 2 \
# --dataset_dir /comp_robot/hongrui/pot_pro/GC10-DET --rotate_degree 30 --epochs 500 \
# --pot_train_mode 1

###POT SEG project
# CUDA_VISIBLE_DEVICES=1 python train_main.py  --backbone resnet --batch_size 4 --testValTrain 4  --base_size 640 \
# --crop_size 640 --dataset CustomPotSeg  --use_txtfile \
# --dataset_dir /home/hongrui/project/metro_pro/dataset/pot/pot_20220108_obvious_defect_0/ --rotate_degree 30 --epochs 200 \
# --pot_train_mode 1 

# CUDA_VISIBLE_DEVICES=6 python train_main.py  --batch_size 8 --testValTrain 4  --base_size 480 \
# --crop_size 480 --dataset CustomPotSeg  --use_txtfile \
# --dataset_dir /home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_1/ --rotate_degree 30 --epochs 200 \
# --pot_train_mode 1 --backbone drn --ramdom_cut_postives --de_ignore_index

# CUDA_VISIBLE_DEVICES=3 python train_main.py  --batch_size 4 --testValTrain 4  --base_size 480 \
# --crop_size 480 --dataset CustomPotSeg  --use_txtfile \
# --dataset_dir /home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_0/ --rotate_degree 30 --epochs 200 \
# --pot_train_mode 1 --backbone drn --ramdom_cut_postives

# CUDA_VISIBLE_DEVICES=2 python train_main.py  --batch_size 8 --testValTrain 4  --base_size 480 --n_classes 2 \
# --crop_size 480 --dataset CustomPotSeg  --use_txtfile \
# --dataset_dir /home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_1/ --rotate_degree 30 --epochs 200 \
# --pot_train_mode 1 --backbone drn --ramdom_cut_postives  --de_ignore_index

CUDA_VISIBLE_DEVICES=2 python train_main.py  --batch_size 8 --testValTrain 4  --base_size 480 \
--crop_size 480 --dataset CustomPotSeg  --use_txtfile \
--dataset_dir /home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/ --rotate_degree 30 --epochs 200 \
--backbone drn --ramdom_cut_postives  --de_ignore_index  --pot_train_mode 5