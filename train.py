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

    #content encoder
    optimizer_ce0 = torch.optim.Adam(model.content_encoder_0.parameters(), lr=0.0001)
    optimizer_ce1 = torch.optim.Adam(model.content_encoder_1.parameters(), lr=0.0001)

    #style encoder
    optimizer_se = torch.optim.Adam(model.style_encoder.parameters(), lr=0.0001)
    #generatore
    optimizer_gen00 = torch.optim.Adam( list(model.encoder_0.parameters()) + list(model.decoder_0.parameters()) , lr=0.0001)
    optimizer_gen01 = torch.optim.Adam( list(model.encoder_0.parameters()) + list(model.decoder_1.parameters()) , lr=0.0001)
    optimizer_gen10 = torch.optim.Adam( list(model.encoder_1.parameters()) + list(model.decoder_0.parameters()) , lr=0.0001)
    optimizer_gen11 = torch.optim.Adam( list(model.encoder_1.parameters()) + list(model.decoder_1.parameters()) , lr=0.0001)

    #discriminator
    optimizer_disc0 = torch.optim.Adam(model.discriminator_0.parameters(), lr=0.0001)
    optimizer_disc1 = torch.optim.Adam(model.discriminator_1.parameters(), lr=0.0001)


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
        optimizer_ce0.zero_grad(); optimizer_ce1.zero_grad(); optimizer_se.zero_grad()
        loss_CE0_reconstruction = l1_distance(outputs["content_encoder_prime"][0], outputs["content_encoder_outputs"][0])
        loss_CE0_reconstruction.backward(retain_graph=True)
        loss_CE1_reconstruction = l1_distance(outputs["content_encoder_prime"][1], outputs["content_encoder_outputs"][1])
        loss_CE1_reconstruction.backward(retain_graph=True)
        loss_Style_reconstruction = l1_distance(outputs["style_encoder_prime"][0], outputs["style_encoder_reconstructed_outputs"][1]) + \
            l1_distance(outputs["style_encoder_prime"][1], outputs["style_encoder_reconstructed_outputs"][0]) 
        loss_Style_reconstruction.backward(retain_graph=True)
        optimizer_ce0.step(); optimizer_ce1.step(); optimizer_se.step()
        # TO DO : Split into 2 for 2 gens
        
        optimizer_gen00.zero_grad(); optimizer_gen01.zero_grad(); optimizer_gen10.zero_grad(); optimizer_gen11.zero_grad()
        loss_gen00 = (mmse_loss(outputs["discriminator_outputs"][0],valid))*1
        loss_gen00.backward(retain_graph=True)
        loss_gen01 = (mmse_loss(outputs["discriminator_outputs"][1],valid))*0.5 
        loss_gen01.backward(retain_graph=True)
        loss_gen10 = (mmse_loss(outputs["discriminator_outputs"][2],valid))*0.5 
        loss_gen10.backward(retain_graph=True)
        loss_gen11 = (mmse_loss(outputs["discriminator_outputs"][3],valid))*0.5 
        loss_gen11.backward(retain_graph=True)
        optimizer_gen00.step(); optimizer_gen01.step(); optimizer_gen10.step(); optimizer_gen11.step()

        optimizer_disc0.zero_grad(); optimizer_disc1.zero_grad()
        loss_disc0 = (adversarial_loss(outputs["discriminator_outputs"][0],fake) + adversarial_loss(outputs["discriminator_outputs"][2],fake))*0.33 + \
        (adversarial_loss(outputs["discriminator_outputs"][4],valid))*0.66 
        loss_disc1 = (adversarial_loss(outputs["discriminator_outputs"][1],fake) + adversarial_loss(outputs["discriminator_outputs"][3],fake))*0.33 + \
        (adversarial_loss(outputs["discriminator_outputs"][5],valid))*0.66 
        loss_disc0.backward(retain_graph=True)
        loss_disc1.backward(retain_graph=True)
        optimizer_disc0.step(); optimizer_disc1.step()

        exit()

if __name__ == "main":
    
    
    train()
