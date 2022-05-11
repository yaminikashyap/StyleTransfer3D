from model import ThreeDsnet
import torch
import random
import numpy
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
from easydict import EasyDict
import numpy as np
import dataset_shapenet as dataset_shapenet
import pointcloud_processor as pointcloud_processor
from argument_parser import parser

def reload_model(device, model_path):
    model = torch.load(model_path, map_location=torch.device(device))
    return model
    

def test_model(model, class_0_file, class_1_file):
    model.eval()
    
    data_0, data_1 = None, None
    
    with torch.no_grad(): 
        
        data = np.load(class_0_file)    
        data_0 = torch.tensor(data).unsqueeze(0)     
        
        data = np.load(class_1_file) #30000 * 3
        data_1 = torch.tensor(data).unsqueeze(0) # 1 * 30000 * 3
        
        shuffled_indices = np.arange(data.shape[0])
        np.random.shuffle(shuffled_indices)

        out_00, out_01, out_10, out_11 = np.zeros((data.shape[0], 3)), np.zeros((data.shape[0], 3)), np.zeros((data.shape[0], 3)), np.zeros((data.shape[0], 3))
        
        for i in range(shuffled_indices.shape[0]//2500):
            start = i*2500
            end = (i+1)*2500
            indices = shuffled_indices[start:end]
            data_0_sample = data_0[:, indices, :]
            data_1_sample = data_1[:, indices, :]
            normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
            data_0_sample[:, :3] = normalization_function(data_0_sample[:, :3])
            data_1_sample[:, :3] = normalization_function(data_1_sample[:, :3])

            data_0_sample = data_0_sample.float()
            data_0_sample = data_0_sample.transpose(2,1).to(device)
            data_1_sample = data_1_sample.float()
            data_1_sample = data_1_sample.transpose(2,1).to(device)

            outputs = model(torch.tile(data_0_sample, (1, 1, 12)), torch.tile(data_1_sample, (1, 1, 12)), train=False)

            out_00[start:end, :] = outputs["00"]
            out_11[start:end, :] = outputs["11"]
            out_01[start:end, :] = outputs["01"]
            out_10[start:end, :] = outputs["10"]

        np.save("./outputs/output00.npy", out_00)
        np.save("./outputs/output11.npy", out_11)
        np.save("./outputs/output01.npy", out_01)
        np.save("./outputs/output10.npy", out_10)

    return 



def test(model, device, class_0_file, class_1_file):
    model = model.to(device)
    test_model(model=model, class_0_file=class_0_file, class_1_file=class_1_file)

if __name__ == "__main__":
    opt = parser()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = reload_model(device, opt.model_path)
    
    print("Found device: ", device)
    test(model=model, device=device, class_0_file=opt.class_0_file, class_1_file=opt.class_1_file)
