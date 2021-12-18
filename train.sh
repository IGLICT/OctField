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

CUDA_VISIBLE_DEVICES=3 python train.py \
    --log_path logs \
    --part_pc_exp_name 'chair'\
    --part_pc_model_epoch 0 \
    --exp_name 'chair1' \
    --model_path model \
    --train_list 'example/chair.txt' \
    --train_list 'example/chair.txt' \
    --epochs 10000 \
    --batch_size 16 \
    --data_path 'example'  \
