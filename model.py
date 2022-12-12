import argparse
import os
import copy
import nni
import cv2
import torch
from torch import nn
import numpy as np
from natsort import natsorted
import imgproc
import torch.optim as optim
import torch.backends.cudnn as cudnn
import config
from image_quality_assessment import PSNR, SSIM
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import ESPCN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr
from utils1 import make_directory
# from test2 import maina

params = {
    'features': 512,
    'lr': 0.001
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)


parser = argparse.ArgumentParser()
parser.add_argument('--train-file', type=str, default="data/91-image_x3.h5")
parser.add_argument('--eval-file', type=str, default="data/Set5_x3.h5")
parser.add_argument('--outputs-dir', type=str, default="output")
parser.add_argument('--weights-file', type=str)
parser.add_argument('--scale', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=123)
args = parser.parse_args()

args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

if not os.path.exists(args.outputs_dir):
    os.makedirs(args.outputs_dir)

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)

model = ESPCN(scale_factor=args.scale).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam([
    {'params': model.first_part.parameters()},
    {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
], lr=args.lr)

train_dataset = TrainDataset(args.train_file)
train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)
eval_dataset = EvalDataset(args.eval_file)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

best_weights = copy.deepcopy(model.state_dict())
best_psnr = 0.0




def train(epoch):
    epoch_losses = AverageMeter()


    model.train()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
        t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))

    # torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

    model.eval()
    epoch_psnr = AverageMeter()

    for data in eval_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
    





def maina(model) -> None:
    # Initialize the super-resolution bsrgan_model
    # g_model = model.__dict__[config.model_arch_name](in_channels=config.in_channels,
    #                                                  out_channels=config.out_channels,
    #                                                  channels=config.channels)
    # g_model = g_model.to(device=config.device)
    # print(f"Build `{config.model_arch_name}` model successfully.")

    # # Load the super-resolution bsrgan_model weights
    # checkpoint = torch.load(config.model_weights_path, map_location=lambda storage, loc: storage)
    # g_model.load_state_dict(checkpoint["state_dict"])
    # print(f"Load `{config.model_arch_name}` model weights "
    #       f"`{os.path.abspath(config.model_weights_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-file', type=str, default="optimized.pth")
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # g_model=model = torch.load(args.eval_file, map_location=lambda storage, loc: storage).to(device)
    g_model= model
    # print(args.eval_file,"is file")
    make_directory(config.sr_dir)

    # Start the verification mode of the bsrgan_model.
    # g_model.eval()

    # Initialize the sharpness evaluation function
    psnr = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=config.device, non_blocking=True)
    ssim = ssim.to(device=config.device, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        gt_image_path = os.path.join(config.gt_dir, file_names[index])

        # print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        gt_y_tensor, gt_cb_image, gt_cr_image = imgproc.preprocess_one_image(gt_image_path, config.device)
        # print("Takes a long time")
        lr_y_tensor, lr_cb_image, lr_cr_image = imgproc.preprocess_one_image(lr_image_path, config.device)

        # Only reconstruct the Y channel image data.
        # print("start")
        with torch.no_grad():
            sr_y_tensor = g_model(lr_y_tensor)

        # print("end")

        # Save image
        sr_y_image = imgproc.tensor_to_image(sr_y_tensor, range_norm=False, half=True)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, gt_cb_image, gt_cr_image])
        sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
        cv2.imwrite(sr_image_path, sr_image * 255.0)

        # Cal IQA metrics
        psnr_metrics += psnr(sr_y_tensor, gt_y_tensor).item()
        ssim_metrics += ssim(sr_y_tensor, gt_y_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files

    print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
          f"SSIM: {avg_ssim:4.4f} [u]")
    return avg_psnr


for epoch in range(args.num_epochs):
    train(epoch)
    psnr = maina(model)
    if(psnr>best_psnr):
        best_psnr=psnr
    nni.report_intermediate_result(psnr)

nni.report_final_result({'default':best_psnr})