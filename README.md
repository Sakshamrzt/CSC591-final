# Course Project

----
## Table Of Contents
- [Description](#description)
- [Training](#commands)
- [Testing](#commands)
- [Team](#team)
----
## Description
Main scripts lies under src directory.

File  | Usage
------------- | -------------
config.py | Config file containing the configuration setting for the experiments
hpo-Evolution.py | Configuration file for evolution HPO tuner
hpo-SMAC.py | Configuration file for evolution SMAC tuner
hpo-TPE.py | Configuration file for evolution TPE tuner
image_quality_assessment.py | Contains image quality operations
imgproc.py | Image processing utilities
model.py | Tuner training file
models.py | File containing the model(ESPCN)
optimize.py | The main file containing the optimization logi
test2.py | Test file
train.py | Training file
utils.py | Contains utilies related to 
----

## Installation
- Run pip3 install -r requirements.txt

## Testing with pretrained model
- The pretrained-model directory contains the pytorch and onnx files for the original and optimized model.
- These models are trained for 3x upscaling. For other upscale factor values new models need to be trained and used for testing.
- These can be used to directly test the effectiveness of the model.
-  Use the command **CUDA_VISIBLE_DEVICES=0 python3 test2.py --eval-file "pretrained-mode/optimized.pth"**. This will generate the files in the results directory and prints the psnr value

## Training
- Run optimize.py file using **CUDA_VISIBLE_DEVICES=0 python3 optimize.py**
- **NOTE** : The work have been tested on GPU's and may have issues on CPU device.
- This will generate the optimized.pth file which can be used further for testing.

## Testing the output
-  If you have generated the new models after using the steps listed in training then you can use : **CUDA_VISIBLE_DEVICES=0 python3 test2.py --eval-file "optimized.pth"**
- This training also generates the optimized.onnx file which can be directly used to run on xgen and speed can be measured there.

## Testing for different scales
- The code is currently configured for 3x upscaling
- For different value of scaling the following needs to be done.
  - Change the upscale_factor in config.py
  - Traing the model using the corresponding dataset. The repository uses dataset configured for 3 x scaling. You can find the appropriate upscaling dataset [here](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD).

## Configuration
- [Training Dataset](https://www.dropbox.com/s/4mv1v4qfjo17zg3/91-image_x3.h5?dl=0)
- [Testing Dataset](https://www.dropbox.com/s/9qlb94in1iqh6nf/Set5_x3.h5?dl=0)
- Upscale value -  Set in the config.py file. Default value is 3.

## References
The pytorch model along with the training/testing part is heavily inspired from the following links.
1. [Pytorch Repository](https://github.com/yjn870/ESPCN-pytorch)
2. [Testing reference](https://github.com/Lornatang/ESPCN-PyTorch)

## Team
Name  | Unity id
------------- | -------------
Saksham Thakur  | sthakur5
Xinning Hui | xhui
---

