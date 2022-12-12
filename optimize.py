import torch
import torch.nn.functional as F
from torch.optim import SGD
import torch.nn as nn
from models import ESPCN
from torch.optim.lr_scheduler import MultiStepLR
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer,DoReFaQuantizer, LsqQuantizer
import os
from train import main, device, train_dataloader
# from eval import evaluate
# from demo import demo
import torch.optim as optim
from test2 import maina
import torch.nn.functional as F

# define the model
model = ESPCN(3).to(device)
# print(model.keys())
model_list = nn.ModuleList([])
# %%
print("Saving the begin")
# define the optimizer and criterion for pre-training
criterion = nn.MSELoss()

optimizer = optim.Adam([
    {'params': model.first_part.parameters()},
    {'params': model.last_part.parameters(), 'lr': 1e-3 * 0.1}
], lr=1e-3)


# pre-train and evaluate the model on MNIST dataset
main(model,"ori")
# main(model)
print("Saving the model")



# %%
# Pruning Model
# -------------
#
# Using L1NormPruner to prune the model and generate the masks.
# Usually, a pruner requires original model and ``config_list`` as its inputs.
# Detailed about how to write ``config_list`` please refer :doc:`compression config specification <../compression/compression_config_list>`.
#
# The following `config_list` means all layers whose type is `Linear` or `Conv2d` will be pruned,
# except the layer named `fc3`, because `fc3` is `exclude`.
# The final sparsity ratio for each layer is 50%. The layer named `fc3` will not be pruned.

config_list = [{
    'sparsity_per_layer': 0.9,
    'op_types': [ 'Conv2d']
    
}, {
    'exclude': True,
    'op_names': ['last_part.0','last_part.1','PixelShuffle']
}]

# %%
# Pruners usually require `model` and `config_list` as input arguments.

from nni.compression.pytorch.pruning import L1NormPruner
pruner = L1NormPruner(model, config_list)

# show the wrapped model structure, `PrunerModuleWrapper` have wrapped the layers that configured in the config_list.
# print(model)

# %%

# compress the model and generate the masks
_, masks = pruner.compress()
# show the masks sparsity
for name, mask in masks.items():
    print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))

# %%
# Speedup the original model with masks, note that `ModelSpeedup` requires an unwrapped model.
# The model becomes smaller after speedup,
# and reaches a higher sparsity ratio because `ModelSpeedup` will propagate the masks across layers.

# need to unwrap the model, if the model is wrapped before speedup
pruner._unwrap_model()

# speedup the model, for more information about speedup, please refer :doc:`pruning_speedup`.
from nni.compression.pytorch.speedup import ModelSpeedup

ModelSpeedup(model, torch.rand(1, 1, 256, 256).to(device), masks).speedup_model()

# %%
# the model will become real smaller after speedup

# %%
# Fine-tuning Compacted Model
# ---------------------------
# Note that if the model has been sped up, you need to re-initialize a new optimizer for fine-tuning.
# Because speedup will replace the masked big layers with dense small ones.
print(model)

# train model again
main(model,"prune")
# evaluate(model)
# demo(model,"prune")
# torch.save(model, "fsrcnn_prune.pt")


# dummy_input = torch.randn(1,1,256,256)
# input_names = ["input"]
# torch.onnx.export(model, dummy_input,  'model_vdsr_pru.onnx', input_names=input_names)

# print("Quantization")
# config_list0 = [ {
#     'quant_types': ['input', 'weight'],
#     'quant_bits': {'input': 8, 'weight': 8},
#     'op_types':['Conv2d', 'Linear']
# }]

# quantizer = LsqQuantizer(model, config_list0, optimizer)
# main(model,"quan")
# evaluate(model)
# demo(model,"quant")
# torch.save(model, "vdsr_prune_quant.pt")
# torch.onnx.export(model, dummy_input,  'model_vdsr_pru_quan.onnx', input_names=input_names)

model_list.append(model)
maina(model)
model_1 = torch.load('ori.pth').to(device)
# In this example, we set the architecture of teacher and student to be the same. It is feasible to set a different teacher architecture.
# model_1.load_state_dict()
model_list.append(model_1)


# class DistillKL(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""
#     def __init__(self, T):
#         super(DistillKL, self).__init__()
#         self.T = T

#     def forward(self, y_s, y_t):
#         p_s = F.log_softmax(y_s/self.T, dim=1)
#         p_t = F.softmax(y_t/self.T, dim=1)
#         loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
#         return loss



# def train_stu( models):
#     global criterion
#     model_s = models[0].to(device).train()
#     model_t = models[-1].to(device).eval()
#     cri_cls = criterion
#     cri_kd = DistillKL(40)


#     for batch_idx, (data, target) in enumerate(train_dataloader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output_s = model_s(data)
#         output_t = model_t(data)

#         loss_cls = cri_cls(output_s, target)
#         loss_kd = cri_kd(output_s, output_t)
#         loss = loss_cls + loss_kd
#         loss.backward()
#         optimizer.step()

os.makedirs('./experiment_data', exist_ok=True)
best_top1 = -1

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

# def train(models):
#     model_s = models[0].train()
#     model_t = models[-1].eval()
#     # cri_kd = DistillKL(4.0)

#     for batch_idx, (data, target) in enumerate(train_dataloader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         y_s = model_s(data)
#         y_t = model_t(data)
#         loss_cri = criterion(y_s, target)

#         # kd loss
#         p_s = F.log_softmax(y_s/4.0, dim=1)
#         p_t = F.softmax(y_t/4.0, dim=1)
#         loss_kd = F.kl_div(p_s, p_t, size_average=False) * (4.0**2) / y_s.shape[0]

#         # total loss
#         loss = loss_cri + loss_kd
#         loss.backward()

# for epoch in range(160):
#     print('# Epoch {} #'.format(epoch), flush=True)
#     train(model_list)
#     # test student only
#     top1 = maina( model_list[0])
#     print(top1,"value")
#     if top1 > best_top1:
#         best_top1 = top1
#         torch.save(model_list[0], 'model_trained_espcn.pth')
#         print('Model trained saved to current dir with loss %f',top1 , flush=True)
# test final model
model.load_state_dict(torch.load('model_trained_espcn.pth'))
# print(model)
# evaluator(model)
dummy_input = torch.randn(1,1, 28, 28).to(device)
torch.onnx.export(model, dummy_input,  'model_espcn_fin.onnx')