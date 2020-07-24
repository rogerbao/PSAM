import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import Generator, MBGenerator, MBPlugGenerator
from model import Discriminator
from PIL import Image
import visdom


class Solver(object):

    def __init__(self, dataset1_loader, dataset2_loader, config):
        self.device_ids = config.device_ids

        # Data loader
        self.dataset1_loader = dataset1_loader
        self.dataset2_loader = dataset2_loader

        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d_train_repeat = config.d_train_repeat
        self.branch_num = config.branch_num
        self.multi_branch_flag = config.multi_branch_flag
        self.branch_size = config.branch_size
        self.branch_combining_mode = config.branch_combining_mode
        self.MB_flag = config.MB_flag
        self.attention_flag = config.attention_flag
        self.only_concat = config.only_concat
        self.im_level = config.im_level
        self.skip_mode = config.skip_mode
        self.skip_act = config.skip_act
        self.m_mode = config.m_mode
        self.with_b = config.with_b
        self.res_mode = config.res_mode

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_mask_l1 = config.lambda_mask_l1
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model
        self.stage1 = config.stage1

        # Test settings
        self.test_model = config.test_model
        self.test_interval = config.test_interval
        self.test_iters = config.test_iters
        self.test_step = config.test_step
        self.test_traverse = config.test_traverse
        self.test_save_path = os.path.join(config.task_name, config.evaluates)

        # Path
        self.log_path = os.path.join(config.task_name, config.log_path)
        self.sample_path = os.path.join(config.task_name, config.sample_path)
        self.model_save_path = os.path.join(config.task_name, config.model_save_path)
        self.result_path = os.path.join(config.task_name, config.result_path)

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.model_save_star = config.model_save_star

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define a generator and a discriminator
        if self.multi_branch_flag:
            self.G = MBPlugGenerator(self.g_conv_dim, [self.c_dim, self.c2_dim], self.g_repeat_num, self.branch_num,
                                     branch_size=self.branch_size, combining_mode=self.branch_combining_mode,
                                     skip_mode=self.skip_mode, skip_act=self.skip_act, m_mode=self.m_mode,
                                     with_b=self.with_b, res_mode=self.res_mode)
        elif self.MB_flag:
            self.G = MBGenerator(self.g_conv_dim, self.c_dim, self.g_repeat_num)

        else:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)

        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        if len(self.device_ids) > 1:
            self.G = torch.nn.DataParallel(self.G, device_ids=self.device_ids)
            self.D = torch.nn.DataParallel(self.D, device_ids=self.device_ids)

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        print("=> using pre-trained model")
        pretrained_dict = torch.load(self.pretrained_model + '_G.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k.find('branchs.2') < 0 and k.find('branchs.1') < 0}
        model_dict = self.G.state_dict()
        model_dict.update(pretrained_dict)
        self.G.load_state_dict(model_dict)

        pretrained_dict = torch.load(self.pretrained_model + '_D.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.find('conv2') < 0}
        model_dict = self.D.state_dict()
        model_dict.update(pretrained_dict)
        self.D.load_state_dict(model_dict)
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x = (x >= 0.5).float()
        # x[x >= 0.5] = 1
        # x[x < 0.5] = 0
        return x

    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA':
            x = F.sigmoid(x)
            predicted = self.threshold(x.data)
            correct = Variable((predicted == y.data).float())
            accuracy = torch.mean(correct, dim=0) * 100.0
        else:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def make_celeb_labels(self, real_c):
        """Generate domain labels for CelebA for debugging/testing.

        if dataset == 'CelebA':
            return single and multiple attribute changes
        elif dataset == 'Both':
            return single attribute changes
        """
        y = [torch.FloatTensor([1, 0, 0]),  # black hair
             torch.FloatTensor([0, 1, 0]),  # blond hair
             torch.FloatTensor([0, 0, 1])]  # brown hair

        fixed_c_list = []

        # single attribute transfer
        for i in range(self.c_dim):
            fixed_c = real_c.clone()
            for c in fixed_c:
                if i < 3:
                    c[:3] = y[i]
                else:
                    c[i] = 0 if c[i] == 1 else 1   # opposite value
            fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        # multi-attribute transfer (H+G, H+A, G+A, H+G+A)
        if self.dataset == 'CelebA':
            for i in range(4):
                fixed_c = real_c.clone()
                for c in fixed_c:
                    if i in [0, 1, 3]:   # Hair color to brown
                        c[:3] = y[2] 
                    if i in [0, 2, 3]:   # Gender
                        c[3] = 0 if c[3] == 1 else 1
                    if i in [1, 2, 3]:   # Aged
                        c[4] = 0 if c[4] == 1 else 1
                fixed_c_list.append(self.to_var(fixed_c, volatile=True))
        return fixed_c_list

    def train(self):

        # Set dataloader
        if self.dataset == 'CelebA':
            self.data_loader = self.dataset1_loader
        else:
            self.data_loader = self.dataset2_loader

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        for i, (images, labels) in enumerate(self.data_loader):
            fixed_x.append(images)
            real_c.append(labels)
            if i == 1:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        real_c = torch.cat(real_c, dim=0)

        if self.dataset == 'CelebA':
            fixed_c_list = self.make_celeb_labels(real_c)
        else:
            fixed_c_list = []
            for i in range(self.c_dim):
                fixed_c = self.one_hot(torch.ones(fixed_x.size(0)) * i, self.c_dim)
                fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (real_x, real_label) in enumerate(self.data_loader):
                
                # Generat fake labels randomly (target domain labels)
                rand_idx = torch.randperm(real_label.size(0))
                fake_label = real_label[rand_idx]

                if self.dataset == 'CelebA':
                    real_c = real_label.clone()
                    fake_c = fake_label.clone()
                else:
                    real_c = self.one_hot(real_label, self.c_dim)
                    fake_c = self.one_hot(fake_label, self.c_dim)

                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_c = self.to_var(real_c)           # input for the generator
                fake_c = self.to_var(fake_c)
                real_label = self.to_var(real_label)   # this is same as real_c if dataset == 'CelebA'
                fake_label = self.to_var(fake_label)
                
                # ================== Train D ================== #

                # Compute loss with real images
                out_src, out_cls = self.D(real_x)
                d_loss_real = - torch.mean(out_src)

                if self.dataset == 'CelebA':
                    d_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls, real_label, size_average=False) / real_x.size(0)
                else:
                    d_loss_cls = F.cross_entropy(out_cls, real_label)

                # Compute classification accuracy of the discriminator
                if (i+1) % self.log_step == 0:
                    accuracies = self.compute_accuracy(out_cls, real_label, self.dataset)
                    log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                    if self.dataset == 'CelebA':
                        print('Classification Acc (Black/Blond/Brown/Gender/Aged): ', end='')
                    else:
                        print('Classification Acc (8 emotional expressions): ', end='')
                    print(log)

                # Compute loss with fake images
                fake_x = self.G(real_x, fake_c)
                fake_x = Variable(fake_x.data)
                out_src, out_cls = self.D(fake_x)
                d_loss_fake = torch.mean(out_src)

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out, out_cls = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1)**2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.data[0]
                loss['D/loss_fake'] = d_loss_fake.data[0]
                loss['D/loss_cls'] = d_loss_cls.data[0]
                loss['D/loss_gp'] = d_loss_gp.data[0]

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x = self.G(real_x, fake_c)
                    rec_x = self.G(fake_x, real_c)

                    # Compute losses
                    out_src, out_cls = self.D(fake_x)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_rec = torch.mean(torch.abs(real_x - rec_x))

                    if self.dataset == 'CelebA':
                        g_loss_cls = F.binary_cross_entropy_with_logits(
                            out_cls, fake_label, size_average=False) / fake_x.size(0)
                    else:
                        g_loss_cls = F.cross_entropy(out_cls, fake_label)

                    # Backward + Optimize
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_fake'] = g_loss_fake.data[0]
                    loss['G/loss_rec'] = g_loss_rec.data[0]
                    loss['G/loss_cls'] = g_loss_cls.data[0]

                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Translate fixed images for debugging
                if (i+1) % self.sample_step == 0:
                    fake_image_list = [fixed_x]
                    for fixed_c in fixed_c_list:
                        fake_image_list.append(self.G(fixed_x, fixed_c))
                    fake_images = torch.cat(fake_image_list, dim=3)
                    save_image(self.denorm(fake_images.data),
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(e+1, i+1)),nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    print('Save model checkpoints')
                    torch.save(self.G.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e+1, i+1)))
                    torch.save(self.D.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_D.pth'.format(e+1, i+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_branch(self):

        # Set dataloader
        if self.dataset == 'CelebA':
            self.data_loader = self.dataset1_loader
            branch_id = 1
        else:
            self.data_loader = self.dataset2_loader
            branch_id = 2

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        for i, (images, labels) in enumerate(self.data_loader):
            fixed_x.append(images)
            real_c.append(labels)
            if i == 0:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        real_c = torch.cat(real_c, dim=0)

        if self.dataset == 'CelebA':
            fixed_c_list = self.make_celeb_labels(real_c)
        else:
            fixed_c_list = []
            for i in range(self.c_dim):
                fixed_c = self.one_hot(torch.ones(fixed_x.size(0)) * i, self.c_dim)
                fixed_c_list.append(self.to_var(fixed_c, volatile=True))

        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start with trained model if exists
        # if self.pretrained_model:
        #     start = int(self.pretrained_model.split('_')[0])
        # else:
        start = 0

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            for i, (real_x, real_label) in enumerate(self.data_loader):
                # Generat fake labels randomly (target domain labels)
                rand_idx = torch.randperm(real_label.size(0))
                fake_label = real_label[rand_idx]

                if self.dataset == 'CelebA':
                    real_c = real_label.clone()
                    fake_c = fake_label.clone()
                else:
                    real_c = self.one_hot(real_label, self.c_dim)
                    fake_c = self.one_hot(fake_label, self.c_dim)

                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_c = self.to_var(real_c)  # input for the generator
                fake_c = self.to_var(fake_c)
                real_label = self.to_var(real_label)  # this is same as real_c if dataset == 'CelebA'
                fake_label = self.to_var(fake_label)

                # ================== Train D ================== #

                # Compute loss with real images
                out_src, out_cls = self.D(real_x)
                d_loss_real = - torch.mean(out_src)

                if self.dataset == 'CelebA':
                    d_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls, real_label, size_average=False) / real_x.size(0)
                else:
                    d_loss_cls = F.cross_entropy(out_cls, real_label)

                # Compute classification accuracy of the discriminator
                if (i + 1) % self.log_step == 0:
                    accuracies = self.compute_accuracy(out_cls, real_label, self.dataset)
                    log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                    if self.dataset == 'CelebA':
                        print('Classification Acc (Black/Blond/Brown/Gender/Aged): ', end='')
                    else:
                        print('Classification Acc (8 emotional expressions): ', end='')
                    print(log)

                # Compute loss with fake images
                fake_x = self.G(real_x, fake_c, branch_id=branch_id, attention_flag = self.attention_flag)
                fake_x = Variable(fake_x.data)
                out_src, out_cls = self.D(fake_x)
                d_loss_fake = torch.mean(out_src)

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out, out_cls = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.data[0]
                loss['D/loss_fake'] = d_loss_fake.data[0]
                loss['D/loss_cls'] = d_loss_cls.data[0]
                loss['D/loss_gp'] = d_loss_gp.data[0]

                # ================== Train G ================== #
                if (i + 1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x = self.G(real_x, fake_c, branch_id, attention_flag = self.attention_flag)
                    rec_x = self.G(fake_x, real_c, branch_id, attention_flag = self.attention_flag)

                    # Compute losses
                    out_src, out_cls = self.D(fake_x)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_rec = torch.mean(torch.abs(real_x - rec_x))

                    if self.dataset == 'CelebA':
                        g_loss_cls = F.binary_cross_entropy_with_logits(
                            out_cls, fake_label, size_average=False) / fake_x.size(0)
                    else:
                        g_loss_cls = F.cross_entropy(out_cls, fake_label)

                    # Backward + Optimize
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_fake'] = g_loss_fake.data[0]
                    loss['G/loss_rec'] = g_loss_rec.data[0]
                    loss['G/loss_cls'] = g_loss_cls.data[0]

                # Print out log info
                if (i + 1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e + 1, self.num_epochs, i + 1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Translate fixed images for debugging
                if (i + 1) % self.sample_step == 0:
                    fake_image_list = [fixed_x]
                    for idx, fixed_c in enumerate(fixed_c_list):
                        fake_image_list.append(self.G(fixed_x, fixed_c, branch_id, attention_flag = self.attention_flag))
                    fake_images = torch.cat(fake_image_list, dim=3)
                    save_image(self.denorm(fake_images.data),
                               os.path.join(self.sample_path, '{}_{}_fake.png'.format(e + 1, i + 1)), nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                # Save model checkpoints
                if (i + 1) % self.model_save_step == 0:
                    print('Save model checkpoints')
                    torch.save(self.G.cpu().state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e + 1, i + 1)))
                    # torch.save(self.D.cpu().state_dict(),
                    #            os.path.join(self.model_save_path, '{}_{}_D.pth'.format(e + 1, i + 1)))
                    self.G.cuda()
                    # self.D.cuda()

            # Decay learning rate
            if (e + 1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        if self.dataset == 'CelebA':
            data_loader = self.dataset1_loader
        else:
            data_loader = self.dataset2_loader

        for i, (real_x, org_c) in enumerate(data_loader):
            real_x = self.to_var(real_x, volatile=True)

            if self.dataset == 'CelebA':
                target_c_list = self.make_celeb_labels(org_c)
            else:
                target_c_list = []
                for j in range(self.c_dim):
                    target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                    target_c_list.append(self.to_var(target_c, volatile=True))

            # Start translations
            fake_image_list = [real_x]
            for target_c in target_c_list:
                fake_image_list.append(self.G(real_x, target_c))
            fake_images = torch.cat(fake_image_list, dim=3)
            save_path = os.path.join(self.result_path, '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))

    def demo(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        if self.dataset == 'CelebA':
            data_loader = self.dataset1_loader
        else:
            data_loader = self.dataset2_loader

        for i, real_x in enumerate(data_loader):
            real_x = self.to_var(real_x, volatile=True)
            print(real_x.size())
            if self.dataset == 'CelebA':
                target_c_list = self.make_celeb_labels(torch.FloatTensor([[1, 0, 0, 0, 1]]))
            else:
                target_c_list = []
                for j in range(self.c_dim):
                    target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                    target_c_list.append(self.to_var(target_c, volatile=True))

            # Start translations
            fake_image_list = [real_x]
            for target_c in target_c_list:
                fake_image_list.append(self.G(real_x, target_c))
            fake_images = torch.cat(fake_image_list, dim=3)
            save_path = os.path.join('demo', '{}_fake.png'.format(i+1))
            save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))

    def evaluate(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        for i in range(self.test_interval[0], self.test_interval[1]+1, self.test_step):
            G_model_name = '{}_{}_G'.format(i, self.test_iters)
            print('Test model {} ...'.format(G_model_name))
            G_path = os.path.join(self.model_save_path, G_model_name + '.pth')
            self.G.load_state_dict(torch.load(G_path))
            self.G.eval()

            data_loader = self.dataset2_loader
            classes = data_loader.dataset.classes
            if not os.path.exists(self.test_save_path):
                os.makedirs(self.test_save_path)
            if not os.path.exists(os.path.join(self.test_save_path, G_model_name)):
                os.makedirs(os.path.join(self.test_save_path, G_model_name))
            for Class in data_loader.dataset.classes:
                if not os.path.exists(os.path.join(self.test_save_path, G_model_name, Class)):
                    os.makedirs(os.path.join(self.test_save_path, G_model_name, Class))

            for i, (real_x, real_label) in enumerate(data_loader):
                real_x = self.to_var(real_x, volatile=True)
                real_c = self.one_hot(real_label, self.c_dim)
                real_c = self.to_var(real_c, volatile=True)


                target_c_list = []
                if self.test_traverse:
                    for j in range(self.c_dim):
                        target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                        target_c_list.append(self.to_var(target_c, volatile=True))

                    # Start translations
                    for j, target_c in enumerate(target_c_list):
                        fake = self.G(real_x, target_c)[0].detach()
                        save_path = os.path.join(self.test_save_path, G_model_name, classes[j], '{}_{}.png'.format(i, j))
                        save_image(self.denorm(fake.data), save_path)
                        print('Translated test images and saved into "{}"..!'.format(save_path))

                else:
                    target_c_list = [real_c]

                    # Start translations
                    for j, target_c in enumerate(target_c_list):
                        fake = self.G(real_x, target_c)[0].detach()
                        save_path = os.path.join(self.test_save_path, G_model_name, classes[real_label[0]], '{}_{}.png'.format(i, classes[real_label[0]]))
                        save_image(self.denorm(fake.data), save_path)
                        print('Translated test images and saved into "{}"..!'.format(save_path))

    def evaluate_branch(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        for i in range(self.test_interval[0], self.test_interval[1]+1, self.test_step):
            G_model_name = '{}_{}_G'.format(i, self.test_iters)
            print('Test model {} ...'.format(G_model_name))
            G_path = os.path.join(self.model_save_path, G_model_name + '.pth')
            self.G.load_state_dict(torch.load(G_path))
            self.G.eval()

            data_loader = self.dataset2_loader
            classes = data_loader.dataset.classes
            if not os.path.exists(os.path.join(self.test_save_path, G_model_name)):
                os.makedirs(os.path.join(self.test_save_path, G_model_name))
            for Class in data_loader.dataset.classes:
                if not os.path.exists(os.path.join(self.test_save_path, G_model_name, Class)):
                    os.makedirs(os.path.join(self.test_save_path, G_model_name, Class))

            for i, (real_x, real_label) in enumerate(data_loader):
                real_x = self.to_var(real_x, volatile=True)
                real_c = self.one_hot(real_label, self.c_dim)
                real_c = self.to_var(real_c, volatile=True)

                target_c_list = []
                if self.test_traverse:
                    for j in range(self.c_dim):
                        target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                        target_c_list.append(self.to_var(target_c, volatile=True))

                    # Start translations
                    for j, target_c in enumerate(target_c_list):
                        fake = self.G(real_x, target_c, branch_id=2, attention_flag=self.attention_flag)[0].detach()
                        save_path = os.path.join(self.test_save_path, G_model_name, classes[j], '{}_{}.png'.format(i, j))
                        save_image(self.denorm(fake.data), save_path)
                        print('Translated test images and saved into "{}"..!'.format(save_path))

                else:
                    target_c_list = [real_c]

                    # Start translations
                    for j, target_c in enumerate(target_c_list):
                        fake = self.G(real_x, target_c, branch_id=2)[0].detach()
                        save_path = os.path.join(self.test_save_path, G_model_name, classes[real_label[0]], '{}_{}.png'.format(i, classes[real_label[0]]))
                        save_image(self.denorm(fake.data), save_path)
                        print('Translated test images and saved into "{}"..!'.format(save_path))

    def vis(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_{}_G.pth'.format(self.test_interval[0], self.test_iters))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()
        vis = visdom.Visdom()

        if self.dataset == 'CelebA':
            data_loader = self.dataset1_loader
        else:
            data_loader = self.dataset2_loader

        for i, (real_x, real_label) in enumerate(data_loader):
            real_x = self.to_var(real_x, volatile=True)
            real_c = self.one_hot(real_label, self.c_dim)
            real_c = self.to_var(real_c, volatile=True)
            if i % 20:
                continue
            target_c_list = []
            for j in range(self.c_dim):
                target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                target_c_list.append(self.to_var(target_c, volatile=True))

            # Start translations
            # image
            # for j, target_c in enumerate(target_c_list):
            #     m, b, f, im = self.G(real_x, target_c, branch_id=2, vis_flag=True)
            #     if j == 0:
            #         vis.image((real_x.cpu().data[0]+1)*127.5)
            #     vis.image((im.cpu().data[0]+1)*127.5)
            # for j, target_c in enumerate(target_c_list):
            #     m, b, f, im = self.G(real_x, target_c, branch_id=2, vis_flag=True)

            #     # if j == 0:
            #     #     vis.image((real_x.cpu().data[0]+1)*127.5)
            #     # vis.image((im.cpu().data[0]+1)*127.5)
            #     res = np.abs(im.cpu().data[0] - real_x.cpu().data[0])
            #     res_ = res.numpy().max(axis=0)
            #     res_ = np.flip(res_, 0)
            #     vis.heatmap(res_)
            for j, target_c in enumerate(target_c_list):
                # m, b, f, im, im_b, im_mf, im_f, im_m, im_fb = self.G(real_x, target_c, branch_id=2, vis_flag=True)
                m, f, im, im_m, im_f = self.G(real_x, target_c, branch_id=2, vis_flag=True)

                # if j == 0:
                #     vis.image((real_x.cpu().data[0]+1)*127.5)
                vis.image((im.cpu().data[0]+1)*127.5)
                # vis.image((im_b.cpu().data[0]+1)*127.5)
                # vis.image((im_mf.cpu().data[0]+1)*127.5)
                # vis.image((im_f.cpu().data[0]+1)*127.5)
                # vis.image((im_m.cpu().data[0]+1)*127.5)
                for w in range(1, 6):
                    im = self.G.act(self.G.smooth(0.2*w*m*f))
                    vis.image((im.cpu().data[0]+1)*127.5)
                # vis.image((im_fb.cpu().data[0]+1)*127.5)
                # res = np.abs(im.cpu().data[0] - real_x.cpu().data[0])
                # res_ = res.numpy().max(axis=0)
                # res_ = np.flip(res_, 0)
                # vis.heatmap(res_)
                # m_ = (m).cpu().data.numpy()[0]
                # m_ = m_.max(axis=0)
                # m_ = np.flip(m_, 0)
                # # vis.heatmap(m_)
                # vis.image(((real_x+1)*127.5*m[:,0:3]).cpu().data[0])
                # vis.image((im.cpu().data[0]+1)*127.5)
                # m__ = (m).cpu().data.numpy()[0]
                # m__ = m__[0]
                # m__ = np.flip(m__, 0)
                # vis.heatmap(m_-m__)
                # f_ = (f).cpu().data.numpy()[0]
                # f_ = f_.max(axis=0)
                # f_ = np.flip(f_, 0)
                # vis.heatmap(f_)
                # mf_ = (m*f).cpu().data.numpy()[0]
                # mf_ = mf_.max(axis=0)
                # mf_ = np.flip(mf_, 0)
                # vis.heatmap(mf_)
                # b_ = (b).cpu().data.numpy()[0]
                # b_ = b_.max(axis=0)
                # b_ = np.flip(b_, 0)
                # vis.heatmap(b_)
                # mfb_ = (m*f+b).cpu().data.numpy()[0]
                # mfb_ = mfb_.max(axis=0)
                # mfb_ = np.flip(mfb_, 0)
                # vis.heatmap(mfb_)
                # print('m:{} f:{} mf:{}, b:{}'.format(m_.max(), f_.max(), mf_.max(), b_.max()))
                # for channel in range(0, 64):
                #     # vis.text('m channel {}'.format(channel))
                #     x = m.cpu().data.numpy()[0][channel]
                #     x = np.rot90(x)
                #     x = np.rot90(x)
                #     vis.heatmap(x)
                #     y = f.cpu().data.numpy()[0][channel]
                #     y = np.rot90(y)
                #     y = np.rot90(y)
                #     vis.heatmap(y)
                #     vis.heatmap(x*y)
                #     z = b.cpu().data.numpy()[0][channel]
                #     z = np.rot90(z)
                #     z = np.rot90(z)
                #     vis.heatmap(z)
                # break
            if i == 3:
                break