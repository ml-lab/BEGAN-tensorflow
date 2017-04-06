from __future__ import print_function

import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

from models import *
from utils import save_image

def next(loader):
    return loader.next()[0].data.numpy()

def norm_img(image):
    return image/255.0

def denorm_img(norm):
    return norm*255.0

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

        self.num_gpu = config.num_gpu
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, self.g_lr * 0.5, name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, self.d_lr * 0.5, name='d_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

    def train(self):
        z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))

        x_fixed = self.data_loader.eval(session=self.sess)
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "k_update": self.k_update,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "measure": self.measure,
                    "k_t": self.k_t,
                })
            result = self.sess.run(fetch_dict)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                g_loss = result['g_loss']
                d_loss = result['d_loss']
                measure = result['measure']
                k_t = result['k_t']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}". \
                      format(step, self.max_step, d_loss, g_loss, measure, k_t))

                x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                cur_measure = np.mean(measure_history)
                self.sess.run([self.g_lr_update, self.d_lr_update])
                #if cur_measure > prev_measure * 0.99:
                #    self.sess.run([self.g_lr_update, self.d_lr_update])
                prev_measure = cur_measure

    def build_model(self):
        _, height, width, channel = int_shape(self.data_loader)
        repeat_num = int(np.log2(height)) - 2

        #self.x = tf.placeholder(tf.float32, [None, height, width, channel], name='y')
        self.x = self.data_loader

        x = norm_img(self.x)

        self.z_D = tf.random_uniform(
                (tf.shape(x)[0], self.conv_hidden_num), minval=-1.0, maxval=1.0)
        self.z_G = tf.random_uniform(
                (tf.shape(x)[0], self.conv_hidden_num), minval=-1.0, maxval=1.0)
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        G_D, self.G_var = GeneratorCNN(
                self.z_D, self.z_num, channel, repeat_num, self.conv_hidden_num, reuse=False)
        G_G, self.G_var = GeneratorCNN(
                self.z_G, self.z_num, channel, repeat_num, self.conv_hidden_num, reuse=True)
        d_out, self.D_var = DiscriminatorCNN(
                tf.concat([G_D, G_G, x], 0), channel, self.z_num, repeat_num, self.conv_hidden_num)
        AE_D, AE_G, AE_x = tf.split(d_out, 3)

        self.G, self.AE_G, self.AE_x = denorm_img(G_G), denorm_img(AE_G), denorm_img(AE_x)

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))
        g_optimizer, d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)

        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - x))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_D - G_D))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.abs(AE_G - G_G))

        d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)
        g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([d_optim, g_optim]):
            self.k_update = tf.assign_add(self.k_t, self.lambda_k * self.balance)
            self.k_t = tf.clip_by_value(self.k_t, 0, 1)

        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.image("AE_G", self.AE_G),
            tf.summary.image("AE_x", self.AE_x),

            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
        ])

    def generate(self, inputs, path, idx=None):
        path = '{}/{}_G.png'.format(path, idx)
        x = self.sess.run(self.G, {self.z_G: inputs})
        save_image(x, path)
        print("[*] Samples saved: {}".format(path))
        return x

    def autoencode(self, inputs, path, idx=None, x_fake=None):
        x_path = '{}/{}_D.png'.format(path, idx)
        x = self.sess.run(self.AE_x, {self.x: inputs})
        save_image(x, x_path)
        print("[*] Samples saved: {}".format(x_path))

        if x_fake is not None:
            x_fake_path = '{}/{}_D_fake.png'.format(path, idx)
            x = self.sess.run(self.AE_x, {self.x: x_fake})
            save_image(x, x_fake_path)
            print("[*] Samples saved: {}".format(x_fake_path))

    def test(self):
        x_fixed = self.data_loader.eval(session=self.sess)
        save_image(x_fixed, '{}/x_fixed_test.png'.format(self.model_dir))
        self.autoencode(x_fixed, self.model_dir, idx="test", x_fake=None)
