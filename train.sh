#!/usr/bin/env sh
python main.py  --task_name 'PSAM-Makeup10k_Eastern-200epoch' --batch_size 12 --mode 'train' \
--dataset2_image_path '../Dataset/Makeup10k_Eastern' --c_dim 5 --c2_dim 5 \
--num_epochs 200 --num_epochs_decay 100 --sample_step 400 --model_save_step 400