from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from scipy.misc import imread, imresize, imsave
from ops import *
import time
import random

class agGAN(object):
    """ The implementation of agGAN """
    def __init__(self,
                 session, # TensorFlow session
                 image_size = 128, # size of input image
                 kernel_size = 3, # size of kernel in convolution and deconcolution
                 batch_size = 10, # mini-batch for training 
                 num_input_channels = 3, # number of channels in input images
                 num_encoder_channels = 64, # number of channels in the first conv layer of encoder
                 num_f_channels = 320, # number of channels of feature representation
                 num_gen_channels = 2048,
                 num_categories = 7, # number of num_categories(age groups) in the training set
                 save_dir = './save', # path to save checkpoints, samples and summary
                 dataset_dir = '', # path to dataset
                 list_file = '',
                 list_file_test = '',
                 mode = 'train',
                 LAMBDA = 10,
                 wgan_gp = True,
                 gen_iters = 0,
                 flip = True
                ):
        self.session = session
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_f_channels = num_f_channels
        self.num_gen_channels = num_gen_channels
        self.num_categories = num_categories
        self.save_dir = save_dir
        self.dataset_dir = dataset_dir
        self.list_file = list_file
        self.list_file_test = list_file_test
        self.mode = mode
        self.num_person = 1876
        self.LAMBDA = LAMBDA
        self.wgan_gp = wgan_gp
        self.gen_iters = gen_iters
        self.flip = flip

        # *********************************input to graph****************************************
        self.image_list = get_dataset(self.dataset_dir, self.list_file)
        self.image_list_test = get_dataset(self.dataset_dir, self.list_file_test)
        assert len(self.image_list) > 0, 'The dataset should not be empty'
        self.data_size = len(self.image_list)
        print('num of images', len(self.image_list))
        
        with tf.name_scope('load_images'):
            #data input flow at training
            path_queue = tf.train.string_input_producer(self.image_list, shuffle=self.mode == "train")

            #number of threads to read image
            num_preprocess_threads = 4
            images_and_labels = []
            
            for _ in range(num_preprocess_threads):
                row = path_queue.dequeue()
                
                fname, label_id, label_age = tf.decode_csv(records=row, record_defaults=[["string"], [""], [""]], field_delim=" ")
                label_id = tf.string_to_number(label_id, tf.int32)
                label_age = tf.string_to_number(label_age, tf.int32)

                #read image
                contents = tf.read_file(fname)
                decode = tf.image.decode_jpeg
                raw_input = decode(contents)
                #scale to 0~1 flip resize to 256x256
                raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
                image = transform(raw_input, self.flip, self.image_size)
                images_and_labels.append([image, label_id, label_age])
            
            self.input_batch, self.label_id_batch, self.label_age_batch = tf.train.batch_join(images_and_labels, batch_size=self.batch_size,
                                                    shapes=[(self.image_size, self.image_size, self.num_input_channels), (), ()],
                                                    capacity=4 * num_preprocess_threads * self.batch_size, allow_smaller_final_batch=True)
            #data placeholder at testing
            
            self.input_batch_test = tf.placeholder(
                                    tf.float32, 
                                    [None, self.image_size, self.image_size, self.num_input_channels],
                                    name='input_batch')

            self.label_id_batch_test = tf.placeholder(
                                    tf.int32, 
                                    [None],
                                    name='label_id_batch')

            self.label_age_batch_test = tf.placeholder(
                                    tf.int32, 
                                    [None],
                                    name='label_age_batch')

        #*************************************build the graph************************************
        with tf.variable_scope('generator'):
            #encoder input image -> generated image, latent representation
            self.G, self.latent, self.logits_id_classify = self.creat_generator(self.input_batch, self.label_age_batch)

            self.G_test, _, _ = self.creat_generator(self.input_batch_test, self.label_age_batch_test, reuse_variables=True, is_training=False)
            
        with tf.variable_scope('discriminator'):
            # discriminator on input image
            self.predict = self.creat_discriminator(self.latent, self.label_age_batch)

        with tf.variable_scope('discriminator_img'):
            # discriminator on input image
            self.predict_real, self.disc_img_real_id, self.disc_img_real_age = self.discriminator_img(self.input_batch)

            # discriminator on G
            self.predict_fake, self.disc_img_fake_id, self.disc_img_fake_age = self.discriminator_img(self.G, reuse_variables=True)
            
        #*************************************loss function**************************************
        with tf.name_scope('total_variation_loss'):
            # total variation to smooth the generated image
            tv_y_size = self.image_size
            tv_x_size = self.image_size
            self.tv_loss = (
            (tf.nn.l2_loss(self.G[:, 1:, :, :] - self.G[:, :self.image_size - 1, :, :]) / tv_y_size) +
            (tf.nn.l2_loss(self.G[:, :, 1:, :] - self.G[:, :, :self.image_size - 1, :]) / tv_x_size)) / self.batch_size

        with tf.name_scope('encoder_loss'):
            self.E_loss_id = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_id_batch, logits=self.logits_id_classify))

            self.loss_E = self.E_loss_id

        with tf.name_scope('generator_loss'):
            #L1 loss
            self.G_loss_L1 = tf.reduce_mean(tf.abs(self.input_batch - self.G)) 
            #L2 loss
            #self.G_loss_L1 = tf.nn.l2_loss(self.input_batch - self.G) / (self.batch_size * 256 * 256 * 3)

            #GAN loss
            EPS = 1e-12
            self.G_loss_GAN = tf.reduce_mean(-tf.log(1 - self.predict + EPS)) 
            self.lambda_e = tf.placeholder(tf.float32, shape=[])
            print(self.lambda_e)

            self.G_fake_loss_d = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predict, labels=tf.ones_like(self.predict_fake)))
            self.G_fake_loss_id = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_id_batch, logits=self.disc_img_fake_id))
            self.G_fake_loss_age = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_age_batch, logits=self.disc_img_fake_age))    

            self.loss_G = self.G_loss_L1 + self.G_fake_loss_d + self.G_fake_loss_id + self.G_fake_loss_age
            
        with tf.name_scope('discriminator_loss'):
            #loss of discriminator
            self.loss_Den = tf.reduce_mean(-tf.log(self.predict + EPS))

        with tf.name_scope('discriminator_img_loss'):
            self.D_real_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.predict_real, labels=tf.ones_like(self.predict_real)))
            self.D_real_loss_id = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_id_batch, logits=self.disc_img_real_id))
            self.D_real_loss_age = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.label_age_batch, logits=self.disc_img_fake_age))    
            
            #fake image
            self.D_fake_loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.predict_fake, labels=tf.zeros_like(self.predict_fake)))

            #loss of discriminator
            #self.loss_D = self.D_real_loss_d + self.D_fake_loss_d
            self.loss_D = self.D_real_loss_d + self.D_fake_loss_d + self.D_real_loss_id + self.D_real_loss_age
        
        #*************************************trainable variables****************************************
        trainable_variables = tf.trainable_variables()
        print('trainable_variables', len(trainable_variables))
        #for var in trainable_variables:
            #print(var.name)

        self.E_variables = [var for var in trainable_variables if 'encoder_' in var.name or 'E_id_' in var.name]
        #variables of generator
        self.G_variables = [var for var in trainable_variables if ('encoder_' in var.name or 'decoder_' in var.name)]
        
        #variables of discriminator
        self.Den_variables = [var for var in trainable_variables if 'layer_' in var.name]
        self.Dimg_variables = [var for var in trainable_variables if 'disc_img_' in var.name]
        print(len(self.E_variables), len(self.G_variables), len(self.Den_variables), len(self.Dimg_variables))
        
        
        #*************************************collect the summary**************************************
        self.G_summary = tf.summary.image('generated_image', self.G)
        self.latent_summary = tf.summary.histogram('latent', self.latent)
        self.predict_summary = tf.summary.histogram('predict', self.predict)
        self.logits_id_classify_summary = tf.summary.histogram('logits_id_classify', self.logits_id_classify)
        self.predict_real_summary = tf.summary.histogram('predict_real', self.predict_real)
        self.disc_img_real_id_summary = tf.summary.histogram('disc_img_real_id', self.disc_img_real_id)
        self.disc_img_real_age_summary = tf.summary.histogram('disc_img_real_age', self.disc_img_real_age)
        self.predict_fake_summary = tf.summary.histogram('predict_fake', self.predict_fake)
        self.disc_img_fake_id_summary = tf.summary.histogram('disc_img_fake_id', self.disc_img_fake_id)
        self.disc_img_fake_age_summary = tf.summary.histogram('disc_img_fake_age', self.disc_img_fake_age)

        
        self.E_loss_id_summary = tf.summary.scalar('E_loss_id', self.E_loss_id)

        self.G_loss_L1_summary = tf.summary.scalar('G_loss_L1', self.G_loss_L1)
        self.G_loss_GAN_summary = tf.summary.scalar('G_loss_GAN', self.G_loss_GAN)
        self.G_fake_loss_d_summary = tf.summary.scalar('G_fake_loss_d', self.G_fake_loss_d)
        self.G_fake_loss_id_summary = tf.summary.scalar('G_fake_loss_id', self.G_fake_loss_id)
        self.G_fake_loss_age_summary = tf.summary.scalar('G_fake_loss_age', self.G_fake_loss_age)

        self.D_real_loss_d_summary = tf.summary.scalar('D_real_loss_d', self.D_real_loss_d)
        self.D_real_loss_id_summary = tf.summary.scalar('D_real_loss_id', self.D_real_loss_id)
        self.D_real_loss_age_summary = tf.summary.scalar('D_real_loss_age', self.D_real_loss_age)
        self.D_fake_loss_d_summary = tf.summary.scalar('D_fake_loss_d', self.D_fake_loss_d)
        
        self.tv_loss_summary = tf.summary.scalar('tv_loss', self.tv_loss)
        self.loss_D_summary = tf.summary.scalar('loss_D', self.loss_D)
        self.loss_G_summary = tf.summary.scalar('loss_G', self.loss_G)
        self.loss_E_summary = tf.summary.scalar('loss_E', self.loss_E)

        #for saving graph and variables
        self.saver = tf.train.Saver(max_to_keep=1)
        
        """
        #************************************run time********************************************
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tf.global_variables_initializer().run()
        step = 0
        start = time.time()
        for i in range(1):
            rrow, rfname, rlabel_id, rlabel_age, rinput_batch, rlabel_id_fake = self.session.run([row, fname, label_id, label_age, self.input_batch, label_id_fake])
            #print(rrow)
            #print(rfname, rlabel_id, rlabel_age)
            #print(rlabel_id_fake)

            #summary = self.summary.eval()
            #step += 1
            #self.writer.add_summary(summary, step)
        print(time.time() - start)
        

        coord.request_stop()
        coord.join(threads)
        """

    def train(self,
              num_epochs, #number of epoch
              learning_rate = 0.002, #initial learning rate
              display_freq = 5000,
              summary_freq = 100,
              save_freq = 5000,
              ):
        #*********************************** optimizer *******************************************************
        global_step = tf.Variable(0, trainable=False, name='global_step')
        global_learning_rate = tf.train.exponential_decay(
            learning_rate = learning_rate, 
            global_step = global_step,
            decay_steps = self.data_size / self.batch_size * 2, #decay leanrning rate each 2 epochs
            decay_rate = 1.0, #learning rate decay (0, 1], 1 means no decay
            staircase = True,
        )
        #print(global_learning_rate.get_shape())
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #update BN moving_mean moving_variance

        beta1 = 0.5  # parameter for Adam optimizer
        with tf.name_scope('discriminator_img_train'):
            with tf.control_dependencies(update_ops):
                D_optimizer = tf.train.AdamOptimizer(
                    learning_rate=global_learning_rate,
                    beta1=beta1
                )
                D_grads_and_vars = D_optimizer.compute_gradients(self.loss_D, var_list=self.Dimg_variables)
                D_train = D_optimizer.apply_gradients(D_grads_and_vars)

        with tf.name_scope('encoder_train'):
            with tf.control_dependencies([D_train]):
                E_optimizer = tf.train.AdamOptimizer(
                    learning_rate=global_learning_rate,
                    beta1=beta1
                )
                E_grads_and_vars = E_optimizer.compute_gradients(self.loss_E, var_list=self.E_variables)
                E_train = E_optimizer.apply_gradients(E_grads_and_vars)
            
        with tf.name_scope('generator_train'):
            with tf.control_dependencies([E_train]):
                G_optimizer = tf.train.AdamOptimizer(
                    learning_rate=global_learning_rate,
                    beta1=beta1
                )
                G_grads_and_vars = G_optimizer.compute_gradients(self.loss_G, var_list=self.G_variables)
                G_train = G_optimizer.apply_gradients(G_grads_and_vars) #must be fetch
        
        #add movingaverage
        #ema = tf.train.ExponentialMovingAverage(0.99)
        #update_losses = ema.apply([self.loss_D, self.loss_E, self.loss_G]) #must be fetch
        incr_global_step = tf.assign(global_step, global_step+1) #must be fetch
        
        #*****************************************collect summary**********************************
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + '/values', var)
        
        for grad, var in D_grads_and_vars + G_grads_and_vars:
            #print(var.name)
            #print(grad)
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        
        with tf.name_scope('parameter_count'):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
        
        #***************************************tensorboard****************************************
        self.global_learning_rate_summary = tf.summary.scalar('global_learning_rate', global_learning_rate)
        self.summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)
        #self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'))
        
        #***************************************training*******************************************
        print('\n Preparing for training...')
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        #initialize the graph
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_batch_per_epoch = int(np.math.ceil(self.data_size / self.batch_size))
        max_steps = num_batch_per_epoch * num_epochs
        print('num_batch_per_epoch', num_batch_per_epoch)
        #(3+1) iters of generator vs 1 iter of discriminator
        
        # epoch iteration
        for epoch in range(num_epochs):
            for batch_ind in range(num_batch_per_epoch):
                start_time = time.time()
                step = global_step.eval()
                def should(freq):
                    return (freq > 0) and ((step+1) % freq == 0 or step+1==max_steps)

                #for i in range(self.gen_iters):
                    #update generator
                    #_ = self.session.run(G_train)
                
                #update generator and discriminator
                fetches = {
                    'G_train': G_train,
                    #'E_train': E_train,
                    'incr_global_step': incr_global_step,
                    'D_err': self.loss_D,
                    'D_real_d_err': self.D_real_loss_d,
                    'D_real_id_err': self.D_real_loss_id,
                    'D_real_age_err': self.D_real_loss_age,
                    'D_fake_d_err': self.D_fake_loss_d,
                    'E_err': self.loss_E,
                    'G_err': self.loss_G,
                    'G_L1_err': self.G_loss_L1,
                    'G_fake_d_err': self.G_fake_loss_d,
                    'G_fake_id_err': self.G_fake_loss_id,
                    'G_fake_age_err': self.G_fake_loss_age,
                    #'G_GAN_err': self.G_loss_GAN,
                    'tv_err': self.tv_loss
                }

                if should(display_freq):
                    fetches['input_batch'] = self.input_batch
                    fetches['G'] = self.G

                if should(summary_freq):
                    fetches['summary'] = self.summary

                #lambda_e = 0 if epoch < 5 else 0.0001
                #lambda_e = 0 if epoch < 3 else 1
                lambda_e = 0
                results = self.session.run(fetches, feed_dict={self.lambda_e: lambda_e})

                if should(display_freq):
                    print("saving display images")
                    """
                    name = '{:06d}_input.png'.format(global_step.eval())
                    self.save_image_batch(results['input_batch'], name)
                    name = '{:06d}_generated.png'.format(global_step.eval())
                    self.save_image_batch(results['G'], name)
                    """
                    name = '{:06d}.png'.format(global_step.eval())
                    self.save_image_pair(results['input_batch'], results['G'], name)

                if should(summary_freq):
                    print("recording summary")
                    self.writer.add_summary(results['summary'], step+1)

                if should(save_freq):
                    print("saving model", checkpoint_dir)
                    self.saver.save(self.session, os.path.join(checkpoint_dir, 'model'), global_step=global_step.eval())

                #estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batch_per_epoch + (num_batch_per_epoch - batch_ind - 1)) * elapse
                if should(40):
                    print('\nEpoch: [%d/%d] Batch: [%d/%d] iter: [%d] G_err=%.4f D_err=%.4f E_err=%.4f' %
                        (epoch+1, num_epochs, batch_ind+1, num_batch_per_epoch, global_step.eval(), results['G_err'], results['D_err'], results['E_err']))
                    print('\tG_L1_err=%.4f G_fake_d_err=%.4f G_fake_id_err=%.4f G_fake_age_err=%.4f' %
                        (results['G_L1_err'], results['G_fake_d_err'], results['G_fake_id_err'], results['G_fake_age_err']))
                    print('\tD_fake_d_err=%.4f D_real_d_err=%.4f D_real_id_err=%.4f D_real_age_err=%.4f' % (
                        results['D_fake_d_err'], results['D_real_d_err'], results['D_real_id_err'], results['D_real_age_err']))
                    print("\t%.2fs/iter Time left: %02d:%02d:%02d lr:%f lambda:%f" %
                          (elapse, int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60, global_learning_rate.eval(), lambda_e))
                    print(self.save_dir)
            #evaluate  
            self.evaluate(epoch+1)

        coord.request_stop()
        coord.join(threads)

    def creat_generator(self, images, labels_age, reuse_variables=False, is_training=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        
        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, num_encoder_channels]
        with tf.variable_scope("encoder_1"):
            output = conv(images, self.num_encoder_channels, stride=2)
            layers.append(output)

        layer_specs = [
            self.num_encoder_channels * 2, # encoder_2: [batch, 128, 128, num_encoder_channels] => [batch, 64, 64, num_encoder_channels*2]
            self.num_encoder_channels * 4, # encoder_3: [batch, 64, 64, num_encoder_channels*2] => [batch, 32, 32, num_encoder_channels*4]
            self.num_encoder_channels * 8, # encoder_4: [batch, 32, 32, num_encoder_channels*4] => [batch, 16, 16, num_encoder_channels*8]
            self.num_encoder_channels * 16,# encoder_5: [batch, 16, 16, num_encoder_channels*8] => [batch, 8,  8, num_encoder_channels*16]
            self.num_encoder_channels * 32,# encoder_6: [batch, 8,  8, num_encoder_channels*16] => [batch,  4,  4, num_encoder_channels*32]
            self.num_encoder_channels * 32,# encoder_7: [batch, 4,  4, num_encoder_channels*32] => [batch,  2,  2, num_encoder_channels*32]
        ]
        
        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(rectified, out_channels, stride=2)
                #output = batchnorm(convolved)
                output = tf.contrib.layers.batch_norm(
                    convolved,
                    scale=True,
                    is_training=is_training,
                    scope='batchnorm',
                    reuse=reuse_variables
                )
                layers.append(output)
                print(output)

        with tf.name_scope('embedding'):
            avg_pool = tf.nn.avg_pool(output, [1, 2, 2, 1], [1, 1, 1, 1], padding="VALID")
            embedding = tf.reshape(
                    avg_pool, [self.batch_size, self.num_encoder_channels * 32])
        print(embedding)

        # id classification layer
        name = 'E_id_classify'
        logits_id_classify = fc(
            input_vector=embedding,
            num_output_length=self.num_person,
            name=name
        )
        print(logits_id_classify)
        
        #decoder
        layer_specs = [
            (self.num_encoder_channels * 32, 0.5), #decoder_8
            (self.num_encoder_channels * 32, 0.5), #decoder_7: [batch, 2, 2, num_encoder_channels*32] => [batch, 4, 4, num_encoder_channels*32 * 2]
            (self.num_encoder_channels * 16, 0.5), #decoder_6:
            (self.num_encoder_channels * 8, 0.0), #decoder_5:
            (self.num_encoder_channels * 4, 0.0),  #decoder_4:
            (self.num_encoder_channels * 2, 0.0),  #decoder_3:
            (self.num_encoder_channels, 0.0),  #decoder_2:
        ]
        num_encoder_layers = len(layers)
        labels_age_one_hot = tf.one_hot(labels_age, self.num_categories, on_value=1.0, off_value=0.0, dtype=tf.float32)
        skip = True
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = concat_label(avg_pool, labels_age_one_hot, self.batch_size)
                    
                else:
                    c = concat_label(layers[-1], labels_age_one_hot, self.batch_size)
                    if skip:
                        #print(c) 
                        input = tf.concat([layers[skip_layer], c], axis=3)
                    else:
                        input = c
                    #print(input)
                
                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = deconv(rectified, out_channels)
                #output = batchnorm(output)
                output = tf.contrib.layers.batch_norm(
                    output,
                    scale=True,
                    is_training=is_training,
                    scope='batchnorm',
                    reuse=reuse_variables
                )

                if dropout > 0.0:
                    output = tf.contrib.layers.dropout(output, keep_prob=1 - dropout, is_training=is_training)
                print(output)
                layers.append(output)
        
        # decoder_1: [batch, 128, 128, num_encoder_channels * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            c = concat_label(layers[-1], labels_age_one_hot, self.batch_size)
            input = tf.concat([layers[0], c], axis=3)
            rectified = tf.nn.relu(input)
            output = deconv(rectified, 3)
            output = tf.tanh(output)
            layers.append(output)
            print(output)
        #print('6', layers[6])
        return output, layers[6], logits_id_classify
    
    def discriminator_img(self, images, reuse_variables=False, is_training=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()

        layers = []

        # [batch, 256, 256, in_channels] => [batch, 128, 128, num_encoder_channels]
        with tf.variable_scope("disc_img_1"):
            output = conv(images, self.num_encoder_channels, stride=2)
            layers.append(output)

        layer_specs = [
            self.num_encoder_channels * 2, # encoder_2: [batch, 128, 128, num_encoder_channels] => [batch, 64, 64, num_encoder_channels*2]
            self.num_encoder_channels * 4, # encoder_3: [batch, 64, 64, num_encoder_channels*2] => [batch, 32, 32, num_encoder_channels*4]
            self.num_encoder_channels * 8, # encoder_4: [batch, 32, 32, num_encoder_channels*4] => [batch, 16, 16, num_encoder_channels*8]
            self.num_encoder_channels * 16,# encoder_5: [batch, 16, 16, num_encoder_channels*8] => [batch, 8,  8, num_encoder_channels*16]
            self.num_encoder_channels * 32,# encoder_6: [batch, 8,  8, num_encoder_channels*16] => [batch,  4,  4, num_encoder_channels*32]
            self.num_encoder_channels * 32,# encoder_7: [batch, 4,  4, num_encoder_channels*32] => [batch,  2,  2, num_encoder_channels*32]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("disc_img_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(rectified, out_channels, stride=2)
                #output = batchnorm(convolved)
                output = tf.contrib.layers.batch_norm(
                    convolved,
                    scale=True,
                    is_training=is_training,
                    scope='batchnorm',
                    reuse=reuse_variables
                )
                layers.append(output)
                print(output)
        
        with tf.name_scope('disc_img_embedding'):
            avg_pool = tf.nn.avg_pool(output, [1, 2, 2, 1], [1, 1, 1, 1], padding="VALID")
            embedding = tf.reshape(
                    avg_pool, [self.batch_size, self.num_encoder_channels * 32])
        print(embedding)

        name = 'disc_img_logit'
        logit = fc(
            input_vector=embedding,
            num_output_length=1,
            name=name
        )

        name = 'disc_img_id'
        disc_img_id = fc(
            input_vector=embedding,
            num_output_length=self.num_person,
            name=name
        )

        name = 'disc_img_age'
        disc_img_age = fc(
            input_vector=embedding,
            num_output_length=self.num_categories,
            name=name
        )

        return logit, disc_img_id, disc_img_age

    def creat_discriminator(self, inputs, labels_age, reuse_variables=False):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()

        layers = []
        labels_age_one_hot = tf.one_hot(labels_age, self.num_categories, on_value=1.0, off_value=0.0, dtype=tf.float32)
        input = concat_label(inputs, labels_age_one_hot, self.batch_size)
        print('1', input)
        
        # layer_1: [batch, 2, 2, 512+7] => [batch, 1, 1, 512]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, 512, stride=2)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 1, 1, 512] => [batch, 512]
        # fully connection layer
        name = 'layer_2'
        output = fc(
            input_vector=tf.reshape(layers[-1], [self.batch_size, 512]),
            num_output_length=512,
            name=name
        )
        layers.append(output)

        # layer_3: [batch, 512] => [batch, 1]
        name = 'layer_3'
        output = fc(
            input_vector=layers[-1],
            num_output_length=1,
            name=name
        )
        sig = tf.sigmoid(output)
        layers.append(sig)

        for l in layers:
            print(l)    
        return layers[-1]

    def discriminator(self, image, is_training=True, reuse_variables=False, enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()

        """
        layer_1: [batch, 128, 128, num_input_channels] => [batch, 64, 64, 64]
        layer_2: [batch, 64, 64, 64] => [batch, 32, 32, 128]
        layer_3: [batch, 32, 32, 128] => [batch, 16, 16, 256]
        layer_4: [batch, 16, 16, 256] => [batch, 8, 8, 512]
        layer_5: [batch, 8, 8, 512] => [batch, 4, 4, 1024]
        layer_6: [batch, 4, 4, 1024] => [batch, 2, 2, 2048]
        """
        num_layers = int(np.log2(self.image_size) - int(self.kernel_size/2)) #6
        print('num_discriminator_layers', num_layers)

        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'D_conv' + str(i)
            current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.kernel_size,
                    name=name
                )
            if enable_bn:
                name = 'D_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
            current = lrelu(current)

        # fully connection layer
        name = 'D_fc'
        current = fc(
            input_vector=tf.reshape(current, [self.batch_size, 2*2*2048]),
            num_output_length=self.num_f_channels,
            name=name
        )
        current = lrelu(current)
        
        name = 'D_logits_age'
        logits_age = fc(
            input_vector=current,
            num_output_length=self.num_categories,
            name=name
        )

        name = 'D_logits_id'
        logits_id = fc(
            input_vector=current,
            num_output_length=self.num_person,
            name=name
        )

        name = 'D_d'
        logits_d = fc(
            input_vector=current,
            num_output_length=1,
            name=name
        )
        
        return logits_id, logits_age, logits_d

    def save_image_batch(self, image_batch, name):
        sample_dir = os.path.join(self.save_dir, 'sample')
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
        #transform the pixel value from -1~1 to 0~1
        images = (image_batch + 1) / 2.0
        frame_size = int(np.sqrt(self.batch_size))
        img_h, img_w = image_batch.shape[1], image_batch.shape[2]
        frame = np.zeros([img_h * frame_size, img_w * frame_size, 3])

        for ind, image in enumerate(images):
            ind_row = ind % frame_size
            ind_col = ind // frame_size
            frame[(img_h*ind_row):(img_h*ind_row+img_h), (img_w*ind_col):(img_w*ind_col+img_w), :] = image

        imsave(os.path.join(sample_dir, name), frame)

    def save_image_pair(self, image_src, image_gen, name):
        sample_dir = os.path.join(self.save_dir, 'sample')
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        #transform the pixel value from -1~1 to 0~1
        images_src = (image_src + 1) / 2.0
        images_gen = (image_gen + 1) / 2.0

        frame_size = int(np.sqrt(self.batch_size))
        img_h, img_w = image_src.shape[1], image_src.shape[2]
        frame = np.zeros([2 * img_h * frame_size, img_w * frame_size, 3])

        for ind, image in enumerate(images_src):
            ind_row = (ind // frame_size) * 2
            ind_col = ind % frame_size
            frame[(img_h*ind_row):(img_h*ind_row+img_h), (img_w*ind_col):(img_w*ind_col+img_w), :] = image

        for ind, image in enumerate(images_gen):
            ind_row = (ind // frame_size) * 2 + 1
            ind_col = ind % frame_size
            frame[(img_h*ind_row):(img_h*ind_row+img_h), (img_w*ind_col):(img_w*ind_col+img_w), :] = image

        imsave(os.path.join(sample_dir, name), frame)

    def evaluate(self, epoch):
        num_images = len(self.image_list_test)
        print('num_images', num_images)

        num_test = 80
        max_steps = int(num_test / 8) #10

        input_batch = np.zeros((64, self.image_size, self.image_size, 3), dtype=np.float32)
        label_id_batch = np.zeros((64,), dtype=np.int32)
        label_age_batch = np.zeros((64,), dtype=np.int32)

        seed = 10
        random.seed(seed)
        random.shuffle(self.image_list_test)
        for i in range(max_steps):
            images_path_labels = self.image_list_test[8*i:8*(i+1)]
            n = 0
            for image_label in images_path_labels:
                image_path = image_label.split()[0]
                id = int(image_label.split()[1])
                age = int(image_label.split()[2])
                #print(image_path, id, age)

                img = imread(image_path)
                img = imresize(img, [self.image_size, self.image_size], interp='nearest')
                img = (img / 255.0) * 2 - 1
                #print(img)
                

                input_batch[7*n:7*(n+1), :, :, :] = img 
                label_id_batch[7*n:7*(n+1)] = id  
                label_age_batch[7*n:7*(n+1)] = [k for k in range(7)]

                n += 1

            #print(input_batch[6,0,0,:])
            #print(input_batch[7,0,0,:])
            #exit(0)

            G = self.session.run(
                self.G_test, 
                feed_dict={
                    self.input_batch_test: input_batch,
                    self.label_id_batch_test: label_id_batch,
                    self.label_age_batch_test: label_age_batch
                }
            )
            name = 'epoch{:02d}_{:06d}.png'.format(epoch, i)
            self.save_test_image_pair(input_batch[0:56], G[0:56], name)
            print("saving images:", name)

    def test(self):
        num_images = self.data_size
        print('num_images', self.data_size)

        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if checkpoint_dir is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            print(checkpoint)
            self.saver.restore(self.session, checkpoint)

        num_test = 80
        max_steps = int(num_test / 8) #10
        
        input_batch = np.zeros((64, self.image_size, self.image_size, 3), dtype=np.float32)
        label_id_batch = np.zeros((64,), dtype=np.int32)
        label_age_batch = np.zeros((64,), dtype=np.int32)

        seed = 10
        random.seed(seed)
        random.shuffle(self.image_list_test)

        input_batch = np.zeros((1, self.image_size, self.image_size, 3), dtype=np.float32)
        label_id_batch = np.zeros((1,), dtype=np.int32)
        label_age_batch = np.zeros((1,), dtype=np.int32)
        image_label = self.image_list_test[0]
        image_path = image_label.split()[0]
        id = int(image_label.split()[1])
        age = int(image_label.split()[2])
        img = imread(image_path)
        img = imresize(img, [self.image_size, self.image_size], interp='nearest')
        img = (img / 255.0) * 2 - 1
        input_batch[0] = img
        label_id_batch[0] = id  
        label_age_batch[0] = age
        G = self.session.run(
                self.G_test, 
                feed_dict={
                    self.input_batch_test: input_batch,
                    self.label_id_batch_test: label_id_batch,
                    self.label_age_batch_test: label_age_batch
                }
            )
        G = (G + 1) / 2.0
        frame = np.zeros([256, 256, 3])
        frame = G[0]
        imsave('1.png', frame)  
        exit(0)    


        for i in range(max_steps):
            images_path_labels = self.image_list_test[8*i:8*(i+1)]
            n = 0
            for image_label in images_path_labels:
                image_path = image_label.split()[0]
                id = int(image_label.split()[1])
                age = int(image_label.split()[2])
                #print(image_path, id, age)

                img = imread(image_path)
                img = imresize(img, [self.image_size, self.image_size], interp='nearest')
                img = (img / 255.0) * 2 - 1
                #print(img)
                

                input_batch[7*n:7*(n+1), :, :, :] = img 
                label_id_batch[7*n:7*(n+1)] = id  
                label_age_batch[7*n:7*(n+1)] = [k for k in range(7)]

                n += 1

            #print(input_batch[6,0,0,:])
            #print(input_batch[7,0,0,:])
            #exit(0)

            G = self.session.run(
                self.G_test, 
                feed_dict={
                    self.input_batch_test: input_batch,
                    self.label_id_batch_test: label_id_batch,
                    self.label_age_batch_test: label_age_batch
                }
            )
            name = 'epoch{:d}_{:06d}.png'.format(50000, i)
            self.save_test_image_pair(input_batch[0:56], G[0:56], name)
            print("saving images:", name)
        """
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        max_steps = int(num_images / self.batch_size)
        print('max_steps', max_steps)
        for i in range(max_steps):
            fetches = {
                'input_batch': self.input_batch,
                'G': self.G
            }
            results = self.session.run(fetches)
            name = '{:06d}.png'.format(i)
            self.save_image_pair(results['input_batch'], results['G'], name)
            print("saving images:", name)

        coord.request_stop()
        coord.join(threads)
        """
        

    def save_test_image_pair(self, image_src, image_gen, name):
        sample_dir = os.path.join(self.save_dir, 'test_sample')
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)

        #transform the pixel value from -1~1 to 0~1
        images_src = (image_src + 1) / 2.0
        images_gen = (image_gen + 1) / 2.0

        frame_size = 8
        img_h = 128
        img_w = 128
        frame = np.zeros([img_h * frame_size, img_w * frame_size, 3])

        for ind_row in range(8):
            ind = ind_row * 7
            img = images_src[ind]
            img = imresize(img, [128, 128], interp='nearest')
            frame[(img_h*ind_row):(img_h*ind_row+img_h), (img_w*0):(img_w*0+img_w), :] = img

        for ind, image in enumerate(images_gen):
            ind_row = (ind // 7)
            ind_col = (ind % 7) + 1
            image = imresize(image, [128, 128], interp='nearest')
            frame[(img_h*ind_row):(img_h*ind_row+img_h), (img_w*ind_col):(img_w*ind_col+img_w), :] = image

        imsave(os.path.join(sample_dir, name), frame)       


        








        


        


        

