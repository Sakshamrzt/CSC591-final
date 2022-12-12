import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import ESPCN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr


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

train_dataset = TrainDataset(args.train_file)
train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)
eval_dataset = EvalDataset(args.eval_file)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def main(model,msg):


    args.outputs_dir = "output/x3"

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True

    torch.manual_seed(args.seed)

    # model = ESPCN(scale_factor=args.scale).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)


    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    print(args.num_epochs," is total epoch")

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))

        model.train()
        epoch_losses = AverageMeter()

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

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

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

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            dummy_input = torch.randn(1,1, 28, 28).to(device)
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model,  msg+'.pth')
            if msg is  "quantization":
                torch.onnx.export(
                model,  # model being run
                dummy_input,  # model inputl
                "espcn"+msg +".onnx")
            else:
                torch.onnx.export(
                    model,  # model being run
                    dummy_input,  # model inputl
                    "espcn"+msg +".onnx",
                    # ,  # where to save the model
                    do_constant_folding=True,    
                    opset_version=9  # XGen supports 11 or 9
                )
            
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))