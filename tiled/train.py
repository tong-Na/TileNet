import time
import os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import random
import numpy as np
import torch

if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print("The number of training images = %d" % dataset_size)

    # 设置随机种子固定
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(
        opt
    )  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    epoch_count = opt.epoch_count
    if opt.continue_train and opt.epoch == "latest":  # 当需要在最新网络参数上继续训练时，读取最新epoch
        log_name = os.path.join(opt.checkpoints_dir, opt.name, "latest_epoch.txt")
        if os.path.exists(log_name):
            with open(log_name, "r") as latest_file:
                latest_epoch = latest_file.read()
                if latest_epoch:
                    epoch_count = int(latest_epoch) + 1

    for epoch in range(
        epoch_count, opt.n_epochs + opt.n_epochs_decay + 1
    ):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.

        if opt.not_gan:
            loss_sum = 0
        counts = 0
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if epoch_iter % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            if opt.not_gan:
                loss_sum += losses["G"]

            if (
                epoch_iter % opt.display_freq == 0
            ):  # display images on visdom and save images to a HTML file
                # save_result = epoch_iter % opt.update_html_freq == 0
                save_result = True
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, counts, save_result
                )

            if (
                epoch_iter % opt.print_freq == 0
            ):  # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data
                )
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses
                    )

            iter_data_time = time.time()
            counts += 1

        if opt.not_gan:
            print("avg loss: %.3f" % (loss_sum / counts))
            avg_loss_file = os.path.join(opt.checkpoints_dir, opt.name, "avg_loss.txt")
            with open(avg_loss_file, "a+") as latest_file:
                latest_file.write(
                    f"epoch: {str(epoch)} avg loss: {(loss_sum / counts):.3f}\n"
                )

        if epoch % opt.save_latest_freq == 0:
            print(
                "saving the latest model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            model.save_networks("latest")
            log_name = os.path.join(opt.checkpoints_dir, opt.name, "latest_epoch.txt")
            with open(log_name, "w+") as latest_file:
                latest_file.write(str(epoch))

        if (
            epoch % opt.save_epoch_freq == 0
        ):  # cache our model every <save_epoch_freq> epochs
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            model.save_networks("latest")
            log_name = os.path.join(opt.checkpoints_dir, opt.name, "latest_epoch.txt")
            with open(log_name, "w+") as latest_file:
                latest_file.write(str(epoch))
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
        )
