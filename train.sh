python train_part.py    --exp_name 'chair' \
    --category 'Chair' \
    --data_path 'example'  \
    --train_dataset 'chair.h5' \
    --loss_weight_kldiv 1e-6 \
    --epoch 10000 \
    --lr 0.001 \
    --lr_decay_every 1000 \
    --lr_decay_by 0.9 \
    --model_path models/

python train.py \
    --log_path logs \
    --exp_name chair \
    --model_path models \
    --data_source data \
    --train_split 'example/chair.txt' \
    --valdt_split 'example/chair.txt' \
    --epochs 10000 \
    --batch_size 16 \
    --voxel_res 8 \
    --loss_weight_center 20.0 \
    --loss_weight_scale 20.0 \
    --loss_weight_type 1.0 \
    --loss_weight_if 20.0 \
    --loss_weight_kldiv 0.05
