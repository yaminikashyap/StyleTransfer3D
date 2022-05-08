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


        output = model(data_0, data_1, train=False).detach().cpu().numpy()
        np.save("./output.npy", output)

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
