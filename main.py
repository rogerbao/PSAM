import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import torch


def str2bool(v):
    return v.lower() in ('true')


def main(config):

    if len(config.device_ids) > 0:
        torch.cuda.set_device(config.device_ids[0])

    # For fast training
    cudnn.benchmark = True
    config.task_name = os.path.join(config.experiment_path, config.task_name)
    # Create directories if not exist
    if not os.path.exists(os.path.join(config.task_name, config.log_path)):
        os.makedirs(os.path.join(config.task_name, config.log_path))
    if not os.path.exists(os.path.join(config.task_name, config.model_save_path)):
        os.makedirs(os.path.join(config.task_name, config.model_save_path))
    if not os.path.exists(os.path.join(config.task_name, config.sample_path)):
        os.makedirs(os.path.join(config.task_name, config.sample_path))
    if not os.path.exists(os.path.join(config.task_name, config.result_path)):
        os.makedirs(os.path.join(config.task_name, config.result_path))
    if not os.path.exists(os.path.join(config.task_name, config.evaluates)):
        os.makedirs(os.path.join(config.task_name, config.evaluates))

    # Data loader
    loader1 = None
    loader2 = None

    if config.mode == 'train':
        loader1 = get_loader(config.dataset1_image_path, config.metadata_path, config.dataset1_crop_size,
                             config.image_size, config.batch_size, 'CelebA', config.mode)
        loader2 = get_loader(config.dataset2_image_path, None, config.dataset2_crop_size,
                             config.image_size, config.batch_size, 'Makeup10k_Eastern', config.mode,
                             config.with_mask)

    elif config.mode == 'evaluate' or config.mode == 'vis':
        loader2 = get_loader(config.dataset2_image_path, None, config.dataset2_crop_size,
                             config.image_size, config.batch_size, config.dataset, 'test')
    else:
        raise Exception('error mode')

    # Solver
    solver = Solver(loader1, loader2, config)

    if config.mode == 'train':
        if config.multi_branch_flag:
            solver.train_branch()
        else:
            solver.train()

    elif config.mode == 'test':
        solver.test()

    elif config.mode == 'evaluate':
        if config.multi_branch_flag:
            solver.evaluate_branch()
        else:
            solver.evaluate()

    elif config.mode == 'vis':
        solver.vis()

    else:
        solver.demo()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device_ids', default=[7])

    # Model hyper-parameters
    parser.add_argument('--c_dim', type=int, default=5)
    parser.add_argument('--c2_dim', type=int, default=5)
    parser.add_argument('--dataset1_crop_size', type=int, default=178)
    parser.add_argument('--dataset2_crop_size', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--lambda_mask_l1', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)
    parser.add_argument('--branch_num', type=int, default=3)
    parser.add_argument('--multi_branch_flag', type=str2bool, default=True)
    parser.add_argument('--MB_flag', type=bool, default=True)
    parser.add_argument('--attention_flag', type=bool, default=False)
    parser.add_argument('--stage1', type=int, default=100000)
    parser.add_argument('--branch_size', type=int, default=7)
    parser.add_argument('--branch_combining_mode', type=str, default='concat')
    parser.add_argument('--only_concat', type=bool, default=False)
    parser.add_argument('--im_level', type=bool, default=False)
    parser.add_argument('--skip_mode', type=str, default='add')
    parser.add_argument('--skip_act', type=str, default='sigmoid')
    parser.add_argument('--res_mode', type=str, default='add')
    parser.add_argument('--m_mode', type=str, default='')
    parser.add_argument('--with_b', type=str, default=True)

    # Training settings
    parser.add_argument('--dataset', type=str, default='Makeup10k_Eastern', choices=['CelebA', 'RaFD', 'Makeup10k_Eastern', 'Multi'])
    parser.add_argument('--dataset1', type=str, default='CelebA')
    parser.add_argument('--dataset2', type=str, default='Makeup10k_Eastern',
                        choices=['CelebA', 'RaFD', 'Makeup10k_Eastern', 'Multi'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=11)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default='')

    # Test settings
    parser.add_argument('--test_model', type=str, default='200000')
    parser.add_argument('--test_interval', nargs='+', type=int, default=[385000, 385000])
    parser.add_argument('--test_iters', type=str, default='400')
    parser.add_argument('--test_step', type=int, default=5)
    parser.add_argument('--test_traverse', type=bool, default=True)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo', 'evaluate', 'vis'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--with_mask', type=bool, default=False)

    # Path
    parser.add_argument('--experiment_path', type=str, default='experiment')
    parser.add_argument('--task_name', type=str, default='PSAM_Makeup10k_Eastern')
    parser.add_argument('--dataset1_image_path', type=str, default='./data/CelebA_nocrop/images')
    parser.add_argument('--dataset2_image_path', type=str, default='../Dataset/Makeup-10k_Eastern')
    parser.add_argument('--metadata_path', type=str, default='../Dataset/list_attr_celeba.txt')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--evaluates', type=str, default='evaluates')

    # Step size
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--sample_step', type=int, default=400)
    parser.add_argument('--model_save_step', type=int, default=400)
    parser.add_argument('--model_save_star', type=int, default=70000)

    config = parser.parse_args()

    print(config)
    main(config)
