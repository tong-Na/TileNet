import os
import torch
from util import html
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from models.net.ffc import FFCResnetBlock
from util.visualizer import save_images


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

    ### 使用未微调的lama时，不注释这段代码
    # model.load_checkpoint(
    #     "~/try-on/tiled/util/lama-main/big-lama/models/best.ckpt"
    # )
    ###

    model.eval()

    web_dir = os.path.join(
        opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.epoch)
    )  # define the website directory

    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s" % (opt.name, opt.phase, opt.epoch),
    )

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader

        print(model.image_paths)

        model.test()
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize,
            use_wandb=opt.use_wandb,
        )
