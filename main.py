from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import glob
import random
import numpy as np
import math
import json
from agGAN import agGAN

flags = tf.app.flags
flags.DEFINE_integer("seed", None, "random seed")
flags.DEFINE_integer("epoch", 50, "number of epochs")
flags.DEFINE_integer("batch_size", 64, "mini-batch for training ")
flags.DEFINE_integer("image_size", 256, "size of input image")
flags.DEFINE_integer("kernel_size", 3, "size of kernel in convolution and deconcolution")
flags.DEFINE_integer("num_input_channels", 3, "number of channels in input images")
flags.DEFINE_integer("num_encoder_channels", 16, "number of channels in the first conv layer of encoder")
flags.DEFINE_integer("num_f_channels", 320, "number of channels of feature representation")
flags.DEFINE_integer("num_categories", 7, "number of num_categories(age groups) in the training set")
flags.DEFINE_integer("num_gen_channels", 2048, "# number of channels of the first deconv layer of generator")
flags.DEFINE_integer("display_freq", 1000, "display_freq")
flags.DEFINE_integer("summary_freq", 100, "summary_freq")
flags.DEFINE_integer("save_freq", 5000, "save_freq")
flags.DEFINE_integer("gen_iters", 0, "gen_iters")
flags.DEFINE_integer("LAMBDA", 10, "Gradient penalty lambda hyperparameter")
flags.DEFINE_string("gpu", "1", "gpu id")
flags.DEFINE_string("dataset_dir", "/media/huangrui/cacd/cacd_mtcnn128", "Path to dataset")
flags.DEFINE_string("list_file", "/media/huangrui/cacd/cacd_mtcnn128_train.txt", "Path to list_file")
flags.DEFINE_string("list_file_test", "/media/huangrui/cacd/cacd_mtcnn128_test.txt", "Path to list_file_test")
flags.DEFINE_string("save_dir", "save/2017615_L1_Esoftmax_Dimg_real_fake", "path to save checkpoints, samples and summary")
flags.DEFINE_string("mode", "train", "train or test")
flags.DEFINE_boolean("flip", True, "flip the image horizontally")

a = flags.FLAGS

def main(_):
    # save the options
    if not os.path.exists(a.save_dir):
        os.mkdir(a.save_dir)
    with open(os.path.join(a.save_dir, 'options.json'), 'w') as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        #with tf.device('/cpu:0'):
            model = agGAN(
                session,
                image_size = a.image_size,
                kernel_size = a.kernel_size,
                batch_size = a.batch_size,
                num_input_channels = a.num_input_channels,
                num_encoder_channels = a.num_encoder_channels,
                num_f_channels = a.num_f_channels,
                num_gen_channels = a.num_gen_channels,
                num_categories = a.num_categories,
                save_dir = a.save_dir,
                dataset_dir = a.dataset_dir,
                list_file = a.list_file,
                list_file_test = a.list_file_test,
                mode = a.mode,
                LAMBDA = a.LAMBDA,
                flip = a.flip
            )

            if a.mode == 'train':
                print('\nTraining mode')
                model.train(num_epochs=a.epoch,
                            learning_rate=0.002,
                            display_freq=a.display_freq,
                            summary_freq=a.summary_freq,
                            save_freq=a.save_freq
                )
            else:
                print('\nTesting mode')
                model.test()


if __name__ == '__main__':
    tf.app.run()