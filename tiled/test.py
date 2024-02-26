import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util import heatmap
import pandas as pd
import numpy as np


try:
    import wandb
except ImportError:
    print(
        'Warning: wandb package cannot be found. The option "--use_wandb" will result in error.'
    )


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.serial_batches = True
    # no flip; comment this line if results on flipped images are needed.
    opt.no_flip = True
    # no visdom display; the test code saves the results to a HTML file.
    opt.display_id = -1
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    # create a model given opt.model and other options
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)

    # initialize logger
    if opt.use_wandb:
        wandb_run = (
            wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt)
            if not wandb.run
            else wandb.run
        )
        wandb_run._label(repo="CycleGAN-and-pix2pix")

    # create a website
    web_dir = os.path.join(
        opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.epoch)
    )  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = "{:s}_iter{:d}".format(web_dir, opt.load_iter)
    print("creating web directory", web_dir)
    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s" % (opt.name, opt.phase, opt.epoch),
    )
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    counts = 0
    if opt.not_gan:
        loss_sum = 0

    data = {"00": "image_id", "11": "landmarks"}
    frame = pd.DataFrame(data, index=[0])
    frame.to_csv("./xxx.csv", mode="w", index=False, header=False)

    for i, data in enumerate(dataset):
        counts += 1
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths

        if opt.not_gan:
            losses = model.get_current_losses()
            loss_sum += losses["G"]

        if i % 50 == 0:  # save images to an HTML file
            print("processing (%04d)-th image... %s" % (i, img_path))
            if opt.not_gan:
                print("avg loss: %.3f" % (loss_sum / (i + 1)))

        if not opt.not_gan:
            save_images(
                webpage,
                visuals,
                img_path,
                aspect_ratio=opt.aspect_ratio,
                width=opt.display_winsize,
                use_wandb=opt.use_wandb,
            )
        # print(model.out_points)
        # print(model.real_points)
        # print(model.image_paths)

        # for keypoint detect model
        # point_tensor = model.out_heatmap.reshape(-1, 2).cpu()
        # tensor to list(keep 4 decimal places)
        # list_line_points = point_tensor.tolist()
        # list_line_points = np.round(list_line_points, 4).tolist()

        # list_line_points = heatmap.find_keypoints(model.out_heatmap)
        # if len(list_line_points) != 11:
        #     print(list_line_points)

        # img_path_list = model.image_paths[0].split("/")
        # img_dir = os.path.join(img_path_list[-2], img_path_list[-1])
        # data = {"name": img_dir, "value": [list_line_points]}
        # frame = pd.DataFrame(data)
        # frame.to_csv("./xxx.csv", mode="a", index=False, header=False)

    if opt.not_gan:
        print("avg loss: %.3f" % (loss_sum / counts))
    webpage.save()  # save the HTML
