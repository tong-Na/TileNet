#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:10:42 2019

@author: zeng
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import glob
import json
import random
import collections
import math
import time
from vgg19 import VGG19
from stn import spatial_transformer_network as transformer

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "pretrain"])
parser.add_argument(
    "--pre_output_dir",
    type=str,
    default=None,
    help="where to put pre-trained output files",
)
parser.add_argument(
    "--output_dir", type=str, default=None, help="where to put output files"
)
parser.add_argument("--seed", type=int)
parser.add_argument(
    "--checkpoint",
    default=None,
    help="directory with checkpoint to resume training from or use for testing",
)
3
parser.add_argument(
    "--max_steps", type=int, help="number of training steps (0 to disable)"
)
parser.add_argument(
    "--max_epochs", type=int, default=200, help="number of training epochs"
)
parser.add_argument(
    "--summary_freq",
    type=int,
    default=100,
    help="update summaries every summary_freq steps",
)
parser.add_argument(
    "--progress_freq",
    type=int,
    default=50,
    help="display progress every progress_freq steps",
)
parser.add_argument(
    "--trace_freq", type=int, default=0, help="trace execution every trace_freq steps"
)
parser.add_argument(
    "--display_freq",
    type=int,
    default=0,
    help="write current training images every display_freq steps",
)
parser.add_argument(
    "--save_freq",
    type=int,
    default=5000,
    help="save model every save_freq steps, 0 to disable",
)

parser.add_argument(
    "--separable_conv",
    action="store_true",
    help="use separable convolutions in the generator",
)
parser.add_argument(
    "--aspect_ratio",
    type=float,
    default=1.0,
    help="aspect ratio of output images (width/height)",
)
parser.add_argument(
    "--lab_colorization",
    action="store_true",
    help="split input image into brightness (A) and color (B)",
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="number of images in batch"
)
parser.add_argument(
    "--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"]
)
parser.add_argument(
    "--ngf",
    type=int,
    default=64,
    help="number of generator filters in first conv layer",
)
parser.add_argument(
    "--ndf",
    type=int,
    default=64,
    help="number of discriminator filters in first conv layer",
)
parser.add_argument(
    "--scale_size",
    type=int,
    default=286,
    help="scale images to this size before cropping to 256x256",
)
parser.add_argument(
    "--flip", dest="flip", action="store_true", help="flip images horizontally"
)
parser.add_argument(
    "--no_flip",
    dest="flip",
    action="store_false",
    help="don't flip images horizontally",
)
parser.set_defaults(flip=True)
parser.add_argument("--cat_nums", type=int, default=10, help="total class number")
parser.add_argument(
    "--lr", type=float, default=0.0002, help="initial learning rate for adam"
)
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument(
    "--l1_weight",
    type=float,
    default=100.0,
    help="weight on L1 term for generator gradient",
)
parser.add_argument(
    "--gan_weight",
    type=float,
    default=1.0,
    help="weight on GAN term for generator gradient",
)
parser.add_argument(
    "--perceptual_weight",
    type=float,
    default=1.0,
    help="weight on GAN term for generator gradient",
)
parser.add_argument(
    "--cls_weight",
    type=float,
    default=1.0,
    help="weight on GAN term for classifier gradient",
)

# export options
parser.add_argument("--output_filetype", default="jpg", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

Examples = collections.namedtuple(
    "Examples", "paths, inputs, targets, classes, count, steps_per_epoch"
)
Model = collections.namedtuple(
    "Model",
    "outputs, labels, cls_loss, perceptual_loss, gen_loss_L1, gen_grads_and_vars, cls_grads_and_vars, train",
)
# Model = collections.namedtuple("Model", "outputs, perceptual_loss, gen_loss_L1, gen_grads_and_vars, train")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def gen_conv(batch_input, out_channels, ker_size, padding="same"):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(
            batch_input,
            out_channels,
            kernel_size=ker_size,
            strides=(2, 2),
            padding="same",
            depthwise_initializer=initializer,
            pointwise_initializer=initializer,
        )
    else:
        return tf.layers.conv2d(
            batch_input,
            out_channels,
            kernel_size=ker_size,
            strides=(2, 2),
            padding=padding,
            kernel_initializer=initializer,
        )


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(
            batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return tf.layers.separable_conv2d(
            resized_input,
            out_channels,
            kernel_size=4,
            strides=(1, 1),
            padding="same",
            depthwise_initializer=initializer,
            pointwise_initializer=initializer,
        )
    else:
        return tf.layers.conv2d_transpose(
            batch_input,
            out_channels,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer=initializer,
        )


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(
        inputs,
        axis=3,
        epsilon=1e-5,
        momentum=0.1,
        training=True,
        gamma_initializer=tf.random_normal_initializer(1.0, 0.02),
    )


def check_image(image):
    assertion = tf.assert_equal(
        tf.shape(image)[-1], 3, message="image must have 3 color channels"
    )
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def tv_loss_fn(y):
    loss_var = tf.reduce_sum(tf.abs(y[:, :, :-1, :] - y[:, :, 1:, :])) + tf.reduce_sum(
        tf.abs(y[:, :-1, :, :] - y[:, 1:, :, :])
    )
    return loss_var


def max_pooling(x):
    return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding="SAME")


def loc_net(x):
    with tf.variable_scope("loc_conv1"):
        theta = gen_conv(x, 20, ker_size=5, padding="valid")
        theta = tf.nn.relu(max_pooling(theta))
    with tf.variable_scope("loc_conv2"):
        theta = gen_conv(theta, 20, ker_size=5, padding="valid")
        theta = tf.nn.relu(max_pooling(theta))
    with tf.variable_scope("fc1"):
        theta = tf.reshape(theta, [1, -1])
        theta = tf.layers.dense(theta, 50)
        theta = tf.nn.relu(theta)
    with tf.variable_scope("fc2"):
        theta = tf.layers.dense(theta, 6)
    return theta


def build_vgg19_model(x, reuse):
    vgg_mean = [103.939, 116.779, 123.68]
    with tf.variable_scope("vgg19", reuse=reuse):
        x = tf.cast(x * 255.0, dtype=tf.float32)

        r, g, b = tf.split(x, 3, 3)
        bgr = tf.concat([b - vgg_mean[0], g - vgg_mean[1], r - vgg_mean[2]], axis=3)
        vgg19 = VGG19(bgr)

        net = vgg19.vgg19_net["conv5_4"]
        return net


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = sorted(glob.glob(os.path.join(a.input_dir, "*.jpg")))
    input_paths_len = len(input_paths)
    print("Train data paths are ready，train data number: {}".format(len(input_paths)))

    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")
    #    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    if a.mode == "train":
        with open("/raid/Guests/zw/pix2pix_all/dataset/train.json", "r") as f:
            label_list = json.load(f)
    else:
        with open("/raid/Guests/zw/pix2pix_all/dataset/train.json", "r") as f:
            label_list = json.load(f)
    print("Train data paths are ready，train data number: {}".format(len(input_paths)))

    decode = tf.image.decode_jpeg

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode

    # TODO    changed

    #    if all(get_name(path).isdigit() for path in input_paths):
    #        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    #    else:
    #        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        #        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        train_list = tf.cast(input_paths, tf.string)
        label_list = tf.cast(label_list, tf.int32)
        input_queue = tf.train.slice_input_producer(
            [train_list, label_list], shuffle=a.mode == "train"
        )
        #        reader = tf.WholeFileReader()
        ##        threads = tf.train.start_queue_runners(sess=sess)
        #        paths, contents = reader.read(input_queue[0])
        paths = input_queue[0]
        classes = input_queue[1]
        contents = tf.read_file(paths)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        #        raw_input = tf.image.resize_images(raw_input, [256, 256])

        assertion = tf.assert_equal(
            tf.shape(raw_input)[2], 3, message="image does not have 3 channels"
        )
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        width = tf.shape(raw_input)[1]  # [height, width, channels]
        a_images = preprocess(raw_input[:, : width // 2, :])
        b_images = preprocess(raw_input[:, width // 2 :, :])

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)

    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(
            r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA
        )

        offset = tf.cast(
            tf.floor(
                tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)
            ),
            dtype=tf.int32,
        )
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(
                r, offset[0], offset[1], CROP_SIZE, CROP_SIZE
            )
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch, classes_batch = tf.train.shuffle_batch(
        [paths, input_images, target_images, classes],
        batch_size=a.batch_size,
        #                                                                                   num_threads= 2,
        capacity=392,
        min_after_dequeue=200,
    )
    steps_per_epoch = int(math.ceil(input_paths_len / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        classes=classes_batch,
        count=input_paths_len,
        steps_per_epoch=steps_per_epoch,
    )


def get_theta(inputs):
    b, h, w, c = inputs.get_shape().as_list()
    #  Create localisation network and convolutional layer
    with tf.variable_scope("stn_theta"):
        #  Create a fully-connected layer with 6 output nodes
        n_fc = 6
        W_fc1 = tf.Variable(tf.zeros([h * w * c, n_fc]), name="W_fc1")

        #  Zoom into the image
        initial = np.array([[0.5, 0, 0], [0, 0.5, 0]])
        initial = initial.astype("float32")
        initial = initial.flatten()

        b_fc1 = tf.Variable(initial_value=initial, name="b_fc1")
        h_fc1 = tf.matmul(tf.zeros([a.batch_size, h * w * c]), W_fc1) + b_fc1

    return h_fc1


# TODO
def cVAE(generator_inputs, generator_outputs_channels, classes):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        #        theta1 = get_theta(generator_inputs)
        theta1 = loc_net(generator_inputs)
        output = transformer(generator_inputs, theta1, (CROP_SIZE, CROP_SIZE))
        output = gen_conv(output, a.ngf, 5)

        layers.append(output)

    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            #    x >= 0    rectified = 0.6x + 0.4|x| = x
            #    x < 0    rectified = 0.6x + 0.4|x| = 0.2x
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, 4)
            output = batchnorm(convolved)
            layers.append(output)

    one_hot = tf.one_hot(classes, 10, 1, 0)
    embedding = tf.reshape(one_hot, (1, 1, 1, 10))
    embedding = tf.cast(embedding, tf.float32)

    layer_specs = [
        (
            a.ngf * 8,
            0.5,
        ),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (
            a.ngf * 8,
            0.5,
        ),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (
            a.ngf * 8,
            0.5,
        ),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (
            a.ngf * 8,
            0.0,
        ),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (
            a.ngf * 4,
            0.0,
        ),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (
            a.ngf * 2,
            0.0,
        ),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (
            a.ngf,
            0.0,
        ),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)  # 8
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                #                input = layers[-1]
                input = tf.concat([layers[-1], embedding], axis=3)
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        #        theta2 = get_theta(output)
        theta2 = loc_net(output)
        output = transformer(output, theta2, (CROP_SIZE, CROP_SIZE))
        layers.append(output)

    return layers[-1]


def create_classifier(inputs):
    n_layers = 5
    layers = []

    # layer_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = gen_conv(inputs, a.ndf, 4)
        rectified = tf.nn.relu(convolved)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 16, 16, ndf * 8]
    # layer_5: [batch, 16, 16, ndf * 4] => [batch, 8, 8, ndf * 8]
    # layer_6: [batch, 8, 8, ndf * 4] => [batch, 4, 4, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = a.ndf * min(2 ** (i + 1), 8)
            #                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = gen_conv(layers[-1], out_channels, 4)
            #                normalized = batchnorm(convolved)
            rectified = tf.nn.relu(convolved)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        convolved = gen_conv(layers[-1], a.cat_nums, 4, padding="valid")
        #            output = tf.sigmoid(convolved)
        layers.append(convolved)

    return layers[-1]


# TODO
def create_model(inputs, targets, classes):
    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = cVAE(inputs, out_channels, classes)

    with tf.name_scope("real_classifier"):
        with tf.variable_scope("classifier"):
            real_logits = create_classifier(targets)

    with tf.name_scope("fake_classifier"):
        with tf.variable_scope("classifier", reuse=True):
            fake_logits = create_classifier(outputs)

    with tf.name_scope("classifier_loss"):
        labels_onehot = tf.one_hot(classes, a.cat_nums, 1, 0)
        classifier_real_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels_onehot, logits=real_logits
            )
        )
        classifier_fake_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels_onehot, logits=fake_logits
            )
        )
        #        classifier_fake_loss = 0.0
        cls_loss = classifier_real_loss + classifier_fake_loss

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0

        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))

        x_real = tf.image.resize_images(targets, size=(224, 224), align_corners=False)
        x_fake = tf.image.resize_images(outputs, size=(224, 224), align_corners=False)

        vgg19_real = build_vgg19_model(x_real, reuse=False)
        vgg19_fake = build_vgg19_model(x_fake, reuse=True)
        perceptual_loss = tf.reduce_mean(tf.abs(vgg19_real - vgg19_fake))

        #        gen_loss = gen_loss_L1 * a.l1_weight + perceptual_loss * a.perceptual_weight
        gen_loss = (
            gen_loss_L1 * a.l1_weight
            + classifier_fake_loss * a.cls_weight
            + perceptual_loss * a.perceptual_weight
        )

    with tf.name_scope("classifier_train"):
        cls_tvars = [
            var for var in tf.trainable_variables() if var.name.startswith("classifier")
        ]
        cls_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        cls_grads_and_vars = cls_optim.compute_gradients(cls_loss, var_list=cls_tvars)
        cls_train = cls_optim.apply_gradients(cls_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([cls_train]):
            gen_tvars = [
                var
                for var in tf.trainable_variables()
                if var.name.startswith("generator")
            ]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(
                gen_loss, var_list=gen_tvars
            )
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([gen_loss_L1, cls_loss, perceptual_loss])
    #    update_losses = ema.apply([ gen_loss_L1, perceptual_loss])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        gen_loss_L1=ema.average(gen_loss_L1),
        perceptual_loss=ema.average(perceptual_loss),
        gen_grads_and_vars=gen_grads_and_vars,
        cls_loss=ema.average(cls_loss),
        cls_grads_and_vars=cls_grads_and_vars,
        outputs=outputs,
        labels=classes,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        #        for kind in ["inputs", "outputs", "targets"]:
        for kind in ["outputs"]:
            filename = name + ".png"
            if step is None:
                filename = "%s" % (filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        #        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")
        index.write("<th>name</th><th>output</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        #        for kind in ["inputs", "outputs", "targets"]:
        for kind in ["outputs"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)
    if a.output_dir is None:
        a.output_dir = "stage1_loc_net" + time.strftime("_%m%d_%H%M")

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets, examples.classes)

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(
                image, size=size, method=tf.image.ResizeMethod.BICUBIC
            )

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(
                tf.image.encode_png,
                converted_inputs,
                dtype=tf.string,
                name="input_pngs",
            ),
            "targets": tf.map_fn(
                tf.image.encode_png,
                converted_targets,
                dtype=tf.string,
                name="target_pngs",
            ),
            "outputs": tf.map_fn(
                tf.image.encode_png,
                converted_outputs,
                dtype=tf.string,
                name="output_pngs",
            ),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("cls_loss", model.cls_loss)
    tf.summary.scalar("perceptual_loss", model.perceptual_loss)
    tf.summary.text("class", tf.as_string(model.labels))

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum(
            [tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()]
        )

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)

        else:
            # training                        #TODO
            start = time.time()
            coord = tf.train.Coordinator()  # 协同启动的线程
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for step in range(max_steps):

                def should(freq):
                    return freq > 0 and (
                        (step + 1) % freq == 0 or step == max_steps - 1
                    )

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["gen_loss_L1"] = model.gen_loss_L1
                    fetches["cls_loss"] = model.cls_loss
                    fetches["perceptual_loss"] = model.perceptual_loss

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(
                        results["summary"], results["global_step"]
                    )

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(
                        results["display"], step=results["global_step"]
                    )
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(
                        run_metadata, "step_%d" % results["global_step"]
                    )

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(
                        results["global_step"] / examples.steps_per_epoch
                    )
                    train_step = (
                        results["global_step"] - 1
                    ) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print(
                        "progress  epoch %d  step %d  image/sec %0.1f  remaining %dm"
                        % (train_epoch, train_step, rate, remaining / 60)
                    )
                    print("gen_loss_L1", results["gen_loss_L1"])
                    print("cls_loss", results["cls_loss"])
                    print("perceptual_loss", results["perceptual_loss"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(
                        sess,
                        os.path.join(a.output_dir, "model"),
                        global_step=sv.global_step,
                    )

                if sv.should_stop():
                    break
            coord.request_stop()  # 停止所有的线程
            coord.join(threads)


main()
