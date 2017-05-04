# original:  InfoGAN/tests/run_mnist_exp.py
# modified: Ekaterina Sutter, May 2017

import sys
sys.path.append('.')

from algos.infogan_trainer import InfoGANTrainer
import argparse

import dateutil
import dateutil.tz
import datetime


import os
from misc.datasets import MnistDataset

from misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli
from misc.utils import mkdir_p
from models.regularized_gan import RegularizedGAN

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='start parser')
    parser.add_argument('-gpu', action='store', default=-1, type=int)
    arg = parser.parse_args()

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "output/logs/mnist"
    root_checkpoint_dir = "output/ckt/mnist"
    batch_size = 128
    # updates_per_epoch = 100
    max_epoch = 1

    exp_name = "mnist_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    dataset = MnistDataset(batch_size)

    latent_spec = [
        (Uniform(62), False),
        (Categorical(10), True),
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True),
    ]

    model = RegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="mnist"
    )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=max_epoch,
        updates_per_epoch=10,  #dataset.train.epochs_size,  # updates_per_epoch,
        info_reg_coeff=1.0,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=2e-4,
        gpu=arg.gpu,
    )

    algo.train()
