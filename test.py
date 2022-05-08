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
best_loss = float('inf')

def reload_model(device):
    model = torch.load("./model85.pt", map_location=torch.device(device))
    # model = ThreeDsnet()
    return model
    

def test_model(model, best_results_dir, classes, batch_size):
    model.eval()
    
    eval_losses = {
        "content_reconstruction": [],
        "style_reconstruction": [],
        "generator": [],
        "discriminator": [],
        "chamfer": [],
        "chamfer_cycle":[]
    }
    
    data_0, data_1 = None, None
    
    with torch.no_grad(): 
        
        data = np.load("bed0.points.ply.npy")    
        data = torch.tensor(data).unsqueeze(0)
        data = data.float()
        data_0 = data.transpose(2,1).to(device)
        
        
        data = np.load("bed1.points.ply.npy")    
        data = torch.tensor(data).unsqueeze(0)
        data = data.float()
        data_1 = data.transpose(2,1).to(device)

        normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        data_0[:, :3] = normalization_function(data_0[:, :3])
        data_1[:, :3] = normalization_function(data_1[:, :3])

        outputs = model(data_0, data_1, train=False).detach().cpu().numpy()

        out_00 = outputs["00"]
        out_11 = outputs["11"]
        out_01 = outputs["01"]
        out_10 = outputs["10"]

        np.save("./output00.npy", out_00)
        np.save("./output11.npy", out_11)
        np.save("./output01.npy", out_01)
        np.save("./output10.npy", out_10)

    return 



def test(model, device, loss_params, batch_size, best_results_dir, classes):
    model = model.to(device)
    # dataloader_eval = dataloaders["eval"]
    test_model(model=model, best_results_dir=best_results_dir, classes=classes, batch_size=batch_size)

if __name__ == "__main__":
    opt = parser()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    model = reload_model(device)
    loss_params = {
        "weight_chamfer" : opt.weight_chamfer,
        "weight_cycle_chamfer" : opt.weight_cycle_chamfer,
        "weight_adversarial" : opt.weight_adversarial,
        "weight_perceptual" : opt.weight_perceptual,
        "weight_content_reconstruction" : opt.weight_content_reconstruction,
        "weight_style_reconstruction" : opt.weight_style_reconstruction
    }
    
    print("Found device: ", device)
    test(model=model, device=device, loss_params=loss_params, batch_size= opt.batch_size, best_results_dir=opt.best_results_dir, classes=[opt.class_0, opt.class_1])
