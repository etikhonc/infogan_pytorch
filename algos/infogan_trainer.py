# original:  InfoGAN/infogan/algos/infogan_trainer.py
# modified: Ekaterina Sutter, May 2017

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from misc.distributions import Bernoulli, Gaussian, Categorical
from models.regularized_gan import RegularizedGAN
import numpy as np
import os.path
from progressbar import ETA, Bar, Percentage, ProgressBar
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.cuda
import torch.optim as optim
import time
import sys

TINY = 1e-8

# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor


class InfoGANTrainer(object):
    def __init__(self,
                 model,
                 batch_size,
                 dataset=None,
                 exp_name="experiment",
                 log_dir="logs",
                 checkpoint_dir="ckt",
                 max_epoch=100,
                 updates_per_epoch=100,
                 snapshot_interval=10000,
                 info_reg_coeff=1.0,
                 discriminator_learning_rate=2e-4,
                 generator_learning_rate=2e-4,
                 gpu=0,
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.info_reg_coeff = info_reg_coeff
        self.discriminator_trainer = None
        self.generator_trainer = None
        x_var = None
        self.log_vars = []
        self.gpu = gpu

        # Optimizer for the generator and discriminator
        self.generator_optimizer = optim.Adam(params=self.model.generator.parameters(),
                                              lr=self.generator_learning_rate, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(params=self.model.discriminator.parameters(),
                                                  lr=self.discriminator_learning_rate, betas=(0.5, 0.999))

        # set gpu device to use
        cudnn.benchmark = True
        if torch.cuda.is_available() and self.gpu < 0:
            print("WARNING: You have a CUDA device which you probably whant to use")

        if gpu > -1:
            torch.cuda.set_device(self.gpu)
            self.model.generator.cuda()
            self.model.discriminator.cuda()
            self.cuda = True
        else:
            self.cuda = False

    @staticmethod
    def reset_grad(params):
        for p in params:
            p.grad.data.zero_()

    def train(self):
        counter = 0

        for epoch in range(self.max_epoch):
            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            # pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
            # pbar.start()

            log_vals = {'G_loss': [], 'D_loss': [], 'lambda': self.info_reg_coeff, 'MI': [], 'Cross_Ent': []}

            for i in range(self.updates_per_epoch):

                print 'epoch %d update %d' % (epoch, i),
                # pbar.update(i)

                # ------------------------------
                # Input tensors
                # ------------------------------
                x, _ = self.dataset.train.next_batch()
                z = self.model.latent_dist.sample_prior(self.batch_size).float()

                # ------------------------------
                # Input variables
                # ------------------------------
                if self.cuda:
                    x = x.cuda()
                    z = z.cuda()
                    
                x_var = Variable(x)
                z_var = Variable(z)
                reg_z = self.model.reg_z(z_var)

                # ------------------------------
                # Update discriminator
                # ------------------------------
                tstart = time.time()

                self.model.discriminator.zero_grad()
                # forward pass: discriminator on REAL data
                real_d, _, _, _ = self.model.discriminate(x_var)
                # forward pass: discriminator on FAKE data
                fake_x, _ = self.model.generate(z_var, is_detach=True)   # do not train G on these labels
                fake_d, _, fake_reg_z_dist_info, _ = self.model.discriminate(fake_x)
                discriminator_loss = - torch.mean(torch.log(real_d + TINY) + torch.log(1. - fake_d + TINY))
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                if self.cuda:
                    D_loss = discriminator_loss.data.cpu().numpy()
                else:
                    D_loss = discriminator_loss.data.numpy()
                log_vals['D_loss'].append(D_loss)
                print 'D_loss %f ' % D_loss,

                # ------------------------------
                # Update: generator
                # ------------------------------
                self.model.generator.zero_grad()
                fake_x, _ = self.model.generate(z_var, is_detach=False)  # NOTE: do not detach
                fake_d, _, fake_reg_z_dist_info, _ = self.model.discriminate(fake_x)
                generator_loss = - torch.mean(torch.log(fake_d + TINY))

                if self.cuda:
                    G_loss = generator_loss.data.cpu().numpy()
                else:
                    G_loss = generator_loss.data.numpy()
                log_vals['G_loss'].append(G_loss)
                print 'G_loss %f ' % G_loss,

                mi_est = 0.
                cross_ent = 0.

                # compute for discrete and continuous codes separately
                # discrete:
                if len(self.model.reg_disc_latent_dist.dists) > 0:
                    disc_reg_z = self.model.disc_reg_z(reg_z)
                    disc_reg_dist_info = self.model.disc_reg_dist_info(fake_reg_z_dist_info)
                    print type(disc_reg_z), type(disc_reg_dist_info),

                    disc_log_q_c_given_x = self.model.reg_disc_latent_dist.logli(disc_reg_z, disc_reg_dist_info)  # log(Q(c|x))
                    disc_log_q_c = self.model.reg_disc_latent_dist.logli_prior(disc_reg_z)
                    # print type(disc_log_q_c_given_x), type(disc_log_q_c),

                    disc_cross_ent = torch.mean(-disc_log_q_c_given_x)
                    disc_ent = torch.mean(-disc_log_q_c)  # H(C)
                    disc_mi_est = disc_ent - disc_cross_ent  # mutual information L_I(G, Q)
                    mi_est += disc_mi_est
                    cross_ent += disc_cross_ent

                    # discriminator_loss -= self.info_reg_coeff * disc_mi_est
                    generator_loss -= self.info_reg_coeff * disc_mi_est

                # if len(self.model.reg_cont_latent_dist.dists) > 0:
                #     cont_reg_z = self.model.cont_reg_z(reg_z)
                #     cont_reg_dist_info = self.model.cont_reg_dist_info(fake_reg_z_dist_info)
                #     cont_log_q_c_given_x = self.model.reg_cont_latent_dist.logli(cont_reg_z, cont_reg_dist_info)  # log(Q(c|x))
                #     cont_log_q_c = self.model.reg_cont_latent_dist.logli_prior(cont_reg_z)
                #     cont_cross_ent = torch.mean(-cont_log_q_c_given_x)
                #     cont_ent = torch.mean(-cont_log_q_c)  # H(C)
                #     cont_mi_est = cont_ent - cont_cross_ent  # mutual information L_I(G, Q)
                #     mi_est += cont_mi_est
                #     cross_ent += cont_cross_ent
                #
                #     # discriminator_loss -= self.info_reg_coeff * cont_mi_est
                #     generator_loss -= self.info_reg_coeff * cont_mi_est

                # if self.cuda:
                #     MI = mi_est.data.cpu().numpy()
                #     Cross_Ent = cross_ent.data.cpu().numpy()
                # else:
                #     MI = mi_est.data.numpy()
                #     Cross_Ent = cross_ent.data.numpy()
                # log_vals['MI'].append(MI)
                # log_vals['Cross_Ent'].append(Cross_Ent)
                # print 'lambda %f, MI %f CrossEnt %f | %0.4fsec' % (self.info_reg_coeff, MI, Cross_Ent,
                #                                                    time.time()-tstart)
                print '| %0.4fsec' % (time.time()-tstart)

                generator_loss.backward()
                self.generator_optimizer.step()
                InfoGANTrainer.reset_grad(self.model.generator.parameters())

                if i % 50 == 0:
                    print 'SAVE'
                    # torch.save(self.model.generator, "%s/G_%d.pth" % (self.checkpoint_dir, i))
                    # torch.save(self.model.discriminator, "%s/D_%d.pth" % (self.checkpoint_dir, i))
                    self.visualize_all_factors(epoch, i)

    # -----------------------------------------------------------------------
    def visualize_all_factors(self, epoch, it):

        # vary discrete variable
        if self.cuda:
            # 128 x 62
            fixed_noncat = np.concatenate([
                np.tile(self.model.nonreg_latent_dist.sample_prior(10).cpu().numpy(), [10, 1]),  # sample from Uniform distribution
                self.model.nonreg_latent_dist.sample_prior(self.batch_size - 100).cpu().numpy()], axis=0)
            # 128 x 12
            fixed_cat = np.concatenate([  # 128 x 12
                np.tile(self.model.reg_latent_dist.sample_prior(10).cpu().numpy(), [10, 1]),
                self.model.reg_latent_dist.sample_prior(self.batch_size - 100).cpu().numpy()], axis=0)
        else:
            # 128 x 62
            fixed_noncat = np.concatenate([
                np.tile(self.model.nonreg_latent_dist.sample_prior(10).numpy(), [10, 1]),  # sample from Uniform distribution
                self.model.nonreg_latent_dist.sample_prior(self.batch_size - 100).numpy()], axis=0)
            # 128 x 12
            fixed_cat = np.concatenate([  # 128 x 12
                np.tile(self.model.reg_latent_dist.sample_prior(10).numpy(), [10, 1]),
                self.model.reg_latent_dist.sample_prior(self.batch_size - 100).numpy()], axis=0)

        offset = 0
        for dist_idx, dist in enumerate(self.model.reg_latent_dist.dists):
            if isinstance(dist, Gaussian):
                # ToDo
                assert dist.dim == 1, "Only dim=1 is currently supported"
                c_vals = []
                for idx in xrange(10):
                    c_vals.extend([-1.0 + idx * 2.0 / 9] * 10)
                c_vals.extend([0.] * (self.batch_size - 100))
                vary_cat = np.asarray(c_vals, dtype=np.float32).reshape((-1, 1))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+1] = vary_cat
                offset += 1
            elif isinstance(dist, Categorical):
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(10):
                    cat_ids.extend([idx] * 10)
                cat_ids.extend([0] * (self.batch_size - 100))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+dist.dim] = lookup[cat_ids]
                offset += dist.dim
            elif isinstance(dist, Bernoulli):
                # ToDo
                assert dist.dim == 1, "Only dim=1 is currently supported"
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(10):
                    cat_ids.extend([int(idx / 5)] * 10)
                cat_ids.extend([0] * (self.batch_size - 100))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+dist.dim] = np.expand_dims(np.array(cat_ids), axis=-1)
                # import ipdb; ipdb.set_trace()
                offset += dist.dim
            else:
                raise NotImplementedError

            # Generate images
            z = torch.from_numpy(np.concatenate([fixed_noncat, cur_cat], axis=1)).float()
            if self.cuda:
                z = z.cuda()
            z_var = Variable(z)
            _, x_dist_info = self.model.generate(z_var)

            if isinstance(self.model.output_dist, Bernoulli):
                img_var = x_dist_info["p"]
            elif isinstance(self.model.output_dist, Gaussian):
                img_var = x_dist_info["mean"]
            else:
                raise NotImplementedError

            if self.cuda:
                samples = img_var.data.cpu().numpy()
            else:
                samples = img_var.data.numpy()

            samples = self.dataset.inverse_transform(samples)
            rows = 10
            samples = np.reshape(samples, [self.batch_size] + list(self.dataset.image_shape))
            samples = samples[:rows * rows, :, :, :]
            imgs = np.reshape(samples, [rows, rows] + list(self.dataset.image_shape))
            stacked_img = []
            for row in xrange(rows):
                row_img = []
                for col in xrange(rows):
                    row_img.append(imgs[row, col, :, :, :])
                stacked_img.append(np.concatenate(row_img, 1))
            imgs = np.concatenate(stacked_img, 0)

            fig = plt.figure(figsize=(4, 4))
            plt.imshow(imgs[:,:,0], cmap='gray')
            plt.axis('off')

            plt.savefig(os.path.join(self.log_dir, 'image_%d_%s_epoch_%03d_it_%03d.png' %
                                     (dist_idx, dist.__class__.__name__, epoch, it)), bbox_inches='tight')
            # np.save(os.path.join(self.log_dir, 'images_epoch_%d_it_%d.png' % (epoch, it)), imgs)
