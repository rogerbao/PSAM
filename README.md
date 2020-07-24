# PSAM
Official PyTorch implementation of "Fine-grained Face Editing via Personalized Spatial-aware Affine Modulation"

~~The code will be released before 2020/07/26.~~

## Requirements

Ubuntu 16.04, with Python 3.6 and PyTorch 0.3+.

You can download Makeup10k Dataset from the [project website](http://www.colalab.org/projects/PSAM).


## Train 
```
python main.py  --task_name 'PSAM-Makeup10k_Eastern-200epoch' --batch_size 12 --mode 'train' \
                --dataset2_image_path '../Dataset/Makeup10k_Eastern' --c_dim 5 --c2_dim 5 \
                --num_epochs 200 --num_epochs_decay 100 --sample_step 400 --model_save_step 400
```

## Test
```
python3 main.py --task_name='PSAM-Makeup10k_Eastern-200epoch' --batch_size=1 --mode='evaluate' \
                --dataset2_image_path='../Dataset/Makeup10k_Eastern' --c_dim=5 --c2_dim=5 \
                --test_iters=400 --test_traverse=True --pretrained_model=''
```