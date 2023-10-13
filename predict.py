import os
import json

import torch
from PIL import Image
from torchvision import transforms

import nibabel as nib
import numpy as np
from model.Madformer import madformer


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # load image
    img_path=r'D:\Codedemo\ADBTS_plus6\NC_057_S_0779.nii'
    data = np.array(nib.load(img_path).get_fdata(), dtype="float32")
    data = np.nan_to_num(data, neginf=0)
    data = normalization(data)
    data = torch.tensor(data)
    data = data[:-1, 4:-5, :-1]
    data = torch.unsqueeze(data, dim=0)
    # +batchsize
    data = torch.unsqueeze(data, dim=0)

    # create model
    model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned").to(device)
    # load model weights
    model_weight_path = "D:\Codedemo\ADBTS_plus6\A_N_best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(data.to(device))[0]).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print(predict_cla)


if __name__ == '__main__':
    main()
