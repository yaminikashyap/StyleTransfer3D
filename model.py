from model import ThreeDsnet
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
from easydict import EasyDict
import dataset_shapenet as dataset_shapenet

classes = ['armchair','straight chair,side chair']
opt = {"data_dir":"/mnt/nfs/work1/mccallum/jbshah/3dsnet/dataset/data/","normalization": "UnitBall", "SVR": True, "sample": True, "number_points": 2500, "shapenet13": True}
dataset_class = dataset_shapenet.ShapeNet
dataset_train = { classes[0]: dataset_class(EasyDict(opt), 'chair', classes[0], train=True),
        classes[1]: dataset_class(EasyDict(opt), 'chair', classes[1], train=True) }

dataloader_train = {}
dataloader_train[classes[0]] = torch.utils.data.DataLoader(
                dataset_train[classes[0]],
                batch_size=4,
                shuffle=True,
            )
dataloader_train[classes[1]] = torch.utils.data.DataLoader(
                dataset_train[classes[1]],
                batch_size=4,
                shuffle=True,
            )

def l1_distance(self, inputs, targets):
    return torch.mean(torch.abs(inputs - targets))

def train():
    model = ThreeDsnet()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()

    #Defining losses
    adversarial_loss = nn.BCELoss()
    mmse_loss = nn.MSELoss()

    #Defining optimizers
    optimizer_D_A = torch.optim.Adam(DnetA.parameters(), lr=0.0001)
    optimizer_D_B = torch.optim.Adam(DnetB.parameters(), lr=0.0001)

    for _, (data_a, data_b) in enumerate(zip(dataloader_train[classes[0]], dataloader_train[classes[1]])):

        data_a = data_a['points']
        # data_a = data_a.view(data_a.shape[0]*data_a.shape[1],3)
        data_a = data_a.transpose(2,1)

        data_b = data_b['points']
        # data_b = data_b.view(data_b.shape[0]*data_b.shape[1],3)
        data_b = data_b.transpose(2,1)

        outputs = model(data_a.to(device),data_b.to(device))

        valid = Variable(Tensor(4, 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(Tensor(4, 1).fill_(0.0), requires_grad=False).to(device)


        #Latent loss
        CE0_reconstruction_loss = l1_distance(outputs["content_encoder_prime"][0], outputs["content_encoder_outputs"][0])
        CE1_reconstruction_loss = l1_distance(outputs["content_encoder_prime"][1], outputs["content_encoder_outputs"][1])
        Style_reconstruction_loss = l1_distance(outputs["style_encoder_prime"][0], outputs["style_encoder_reconstructed_outputs"][1]) + \
            l1_distance(outputs["style_encoder_prime"][1], outputs["style_encoder_reconstructed_outputs"][0]) 

        # TO DO : Split into 2 for 2 gens
        
        loss_gen0 = (mmse_loss(outputs["discriminator_outputs"][0],valid) + mmse_loss(outputs["discriminator_outputs"][3],valid))*0.33 + \
        (mmse_loss(outputs["discriminator_outputs"][4],fake))*0.66
        loss_gen1 = (mmse_loss(outputs["discriminator_outputs"][1],valid) + mmse_loss(outputs["discriminator_outputs"][2],valid))*0.33 + \
        (mmse_loss(outputs["discriminator_outputs"][5],fake))*0.66

        loss_disc= (adversarial_loss(outputs["discriminator_outputs"][0],fake) + adversarial_loss(outputs["discriminator_outputs"][1],fake) +
        adversarial_loss(outputs["discriminator_outputs"][2],fake) + adversarial_loss(outputs["discriminator_outputs"][3],fake))*0.33 + \
        (adversarial_loss(outputs["discriminator_outputs"][4],valid) + adversarial_loss(outputs["discriminator_outputs"][5],valid))*0.66
        loss_disc.backward(retain_graph=True)
        # exit()

if __name__ == "main":
    
    
    train()
