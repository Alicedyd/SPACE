import os
import time

try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    from torch.utils.tensorboard import SummaryWriter
from data import create_dataloader
from networks.trainer import Trainer
from options import TrainOptions
from validate import validate_and_save_results
import shutil
import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)  # 控制 torchvision.transforms 等
    np.random.seed(seed)  # 控制 numpy 操作
    torch.manual_seed(seed)  # 控制 torch 操作
    torch.cuda.manual_seed_all(seed)  # 多卡时也生效
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # set_seed(42)
    opt = TrainOptions().parse()

    model = Trainer(opt)

    data_loader = create_dataloader(opt)

    results_file = os.path.join(opt.checkpoints_dir, opt.name, "validation_results.csv")
    loss_log_path = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")

    config_path = "../val_configs/Chameleon.yaml"

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    start_time = time.time()
    print("Length of data loader: %d" % (len(data_loader)))

    for epoch in range(opt.niter):
        for i, data in enumerate(data_loader):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % 1000 == 0:
                print(
                    "Train loss: {} at step: {}".format(model.loss, model.total_steps)
                )
                with open(loss_log_path, "a") as f:
                    f.write(f"{model.total_steps},{model.loss:.6f}\n")
                train_writer.add_scalar("loss", model.loss, model.total_steps)
                print("Iter time: ", ((time.time() - start_time) / model.total_steps))

            if (
                model.total_steps >= 60000 and model.total_steps % 5000 == 0
            ):  # save models at these iters
                # if model.total_steps % 100 == 0:
                model.save_networks("model_iters_%s.pth" % model.total_steps)
                with open(loss_log_path, "a") as f:
                    f.write(f"{model.total_steps},{model.loss:.6f}\n")
                # Run validation with sequential evaluation and thresholds
                model.eval()

                # Use the new sequential validation function instead of validate_and_save_results
                validate_and_save_results(
                    model.model,
                    config_path,
                    results_file,
                    iteration=model.total_steps,
                    epoch=epoch,
                    max_sample=1000,
                    batch_size=128,
                    gpu_id=opt.gpu_ids[0],
                    arch=opt.arch,
                    cropSize=opt.cropSize,
                    is_crop=not opt.is_resize,
                )
                model.train()

        model.finalize_epoch()
        print("saving the model at the end of epoch %d" % (epoch))
        model.save_networks("model_epoch_best.pth")
        model.save_networks("model_epoch_%s.pth" % epoch)
