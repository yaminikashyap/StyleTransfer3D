from model_optimized import ThreeDsnet
import torch
import random
import numpy
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
from easydict import EasyDict
import dataset_shapenet as dataset_shapenet
from argument_parser import parser
best_loss = float('inf')

def prepare_data(dataset_class, family, classes, data_dir, batch_size):
    data_options = {
        "data_dir":data_dir,
        "normalization": "BoundingBox",
        "SVR": False, 
        "sample": True, 
        "number_points": 2500
    }
    dataloaders = {}
    dataloader_train = {}
    
    dataset_train = { 
        classes[0]: dataset_class(EasyDict(data_options), family, classes[0], train=True),
        classes[1]: dataset_class(EasyDict(data_options), family, classes[1], train=True) 
    }
    dataloader_train[classes[0]] = torch.utils.data.DataLoader(
                    dataset_train[classes[0]],
                    batch_size=batch_size,
                    shuffle=True,
                )
    dataloader_train[classes[1]] = torch.utils.data.DataLoader(
                    dataset_train[classes[1]],
                    batch_size=batch_size,
                    shuffle=True,
                )
    
    dataloader_eval = {}
    dataset_eval = { 
        classes[0]: dataset_class(EasyDict(data_options), family, classes[0], train=False),
        classes[1]: dataset_class(EasyDict(data_options), family, classes[1], train=False) 
    }
    dataloader_eval[classes[0]] = torch.utils.data.DataLoader(
                    dataset_eval[classes[0]],
                    batch_size=batch_size,
                    shuffle=False,
                )
    dataloader_eval[classes[1]] = torch.utils.data.DataLoader(
                    dataset_eval[classes[1]],
                    batch_size=batch_size,
                    shuffle=False,
                )
    
    dataloaders["train"] = dataloader_train
    dataloaders["eval"] = dataloader_eval
    return dataloaders

def prepare_model():
    model = ThreeDsnet()
    return model

def calculate_losses(outputs, loss_params, data_0, data_1, batch_size, train=True):
    adversarial_loss = adversarial_loss_cal
    reconstruction_loss = l1_distance
    chamfer_loss = chamfer_dist

    #Content Encoder losses
    loss_CE0_reconstruction = loss_params["weight_content_reconstruction"] * reconstruction_loss(outputs["content_encoder_prime"][0], outputs["content_encoder_outputs"][0])
    
    loss_CE1_reconstruction = loss_params["weight_content_reconstruction"] * reconstruction_loss(outputs["content_encoder_prime"][1], outputs["content_encoder_outputs"][1])
    
    #print("CE0 ", loss_CE0_reconstruction)
    #print("CE1 ", loss_CE1_reconstruction)
    if train:
        loss_CE0_reconstruction.backward(retain_graph=True)
        loss_CE1_reconstruction.backward(retain_graph=True)

    #Style Encoder losses
    loss_style_reconstruction = loss_params["weight_style_reconstruction"] * (reconstruction_loss(outputs["style_encoder_primes"][0], outputs["style_encoder_reconstructed_outputs"][1]) + reconstruction_loss(outputs["style_encoder_primes"][1], outputs["style_encoder_reconstructed_outputs"][0]))

    #print("Style ", loss_style_reconstruction)
    
    if train:
        loss_style_reconstruction.backward(retain_graph=True)

    #Adversarial losses
    #disc0 -> should identify out_10, data0
    #disc1 -> should identify out_01, data1
    
    adversarial_loss_0 = loss_params["weight_adversarial"]*adversarial_loss(outputs["discriminator_outputs"][2], outputs["discriminator_outputs"][4])
    adversarial_loss_1 = loss_params["weight_adversarial"]*adversarial_loss(outputs["discriminator_outputs"][1], outputs["discriminator_outputs"][5])
    
    #print("Adv0 ", adversarial_loss_0)
    #print("Adv1 ", adversarial_loss_1)
    if train:
        adversarial_loss_0.backward(retain_graph=True)
        adversarial_loss_1.backward(retain_graph=True)

    #Chamfer loss - Identity
    chamfer_loss_00 = loss_params["weight_chamfer"]*chamfer_loss(data_0, outputs["reconstructed_outputs"][0])
    chamfer_loss_11 = loss_params["weight_chamfer"]*chamfer_loss(data_1, outputs["reconstructed_outputs"][1])
    
    #print("Chamfer00 ", chamfer_loss_00)
    #print("Chamfer11 ", chamfer_loss_11)

    if train:
        chamfer_loss_00.backward(retain_graph=True)
        chamfer_loss_11.backward(retain_graph=True)
    
    #Chamfer loss - Cycle
    chamfer_loss_010 = loss_params["weight_cycle_chamfer"]*chamfer_loss(data_0, outputs["cycle_reconstructed_outputs"][0]["points_3"].view(batch_size, -1, 3))
    chamfer_loss_101 = loss_params["weight_cycle_chamfer"]*chamfer_loss(data_1, outputs["cycle_reconstructed_outputs"][1]["points_3"].view(batch_size, -1, 3))
    

    #print("Chamfer010 ", chamfer_loss_010)
    #print("Chamfer101 ", chamfer_loss_101)
    if train:
        chamfer_loss_010.backward(retain_graph=True)
        chamfer_loss_101.backward(retain_graph=True)
    
    return [(loss_CE0_reconstruction + loss_CE1_reconstruction).detach().cpu().numpy(), loss_style_reconstruction.detach().cpu().numpy(), (adversarial_loss_0 + adversarial_loss_1).detach().cpu().numpy(), (chamfer_loss_00 + chamfer_loss_11).detach().cpu().numpy(), (chamfer_loss_010 + chamfer_loss_101).detach().cpu().numpy()]
    
    
def optimizer_zero_grad(optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad()
        
def optimizer_step(optimizers):
    for optimizer in optimizers:
        optimizer.step()

def prepare_optimizers(model, generator_lrate, discriminator_lrate):
    
    #generator optimizers
    optimizer_ce0 = torch.optim.Adam(model.content_encoder_0.parameters(), lr=generator_lrate)
    optimizer_ce1 = torch.optim.Adam(model.content_encoder_1.parameters(), lr=generator_lrate)
    optimizer_se = torch.optim.Adam(model.style_encoder.parameters(), lr=generator_lrate)
    optimizer_de0 = torch.optim.Adam(model.decoder_0.parameters(), lr=generator_lrate)
    optimizer_de1 = torch.optim.Adam(model.decoder_1.parameters(), lr=generator_lrate)
    
    #discriminator optimizers
    optimizer_disc0 = torch.optim.Adam(model.discriminator_0.parameters(), lr=discriminator_lrate)
    optimizer_disc1 = torch.optim.Adam(model.discriminator_1.parameters(), lr=discriminator_lrate)

    return [optimizer_ce0, optimizer_ce1, optimizer_se, optimizer_de0, optimizer_de1, optimizer_disc0, optimizer_disc1]

def l1_distance(inputs, targets):
    return torch.mean(torch.abs(inputs - targets))

def chamfer_dist(reconstructed_points, target_points):
    reconstructed_points = torch.transpose(reconstructed_points, 1, 2)
    chamfer_loss = 0
    number_points = 0
    i = 0
    for batch_idx in range(reconstructed_points.shape[0]):
        batch_points = int(target_points.shape[1]*0.05)
        number_points += batch_points
        indices = numpy.arange(target_points.shape[1]).tolist()
        sampled_points = random.sample(indices, batch_points)
        for point in target_points[batch_idx][sampled_points]:
            diff = torch.square(reconstructed_points[batch_idx].to(device) - point)
            diff = torch.sum(diff, dim = 1)
            diff = torch.min(diff)
            chamfer_loss += diff
            i+=1
    reconstructed_points = torch.transpose(reconstructed_points, 1, 2)
    return torch.div(chamfer_loss, number_points)

def adversarial_loss_cal(generated, real):
    generated_ = torch.clamp(1 - generated, min=0.0001)
    real_ = torch.clamp(real, min=0.0001)
    return torch.mean(torch.log(generated_)) + torch.mean(torch.log(real_))

def train_epoch(dataloader_train, model, optimizers, device, loss_params, batch_size, classes):
    
    batch_count = 0
    content_reconstruction_loss = 0
    style_reconstruction_loss = 0
    adversarial_loss = 0
    chamfer_loss = 0
    chamfer_cycle_loss = 0
    
    data_0, data_1 = None, None
    for _, (batch_a, batch_b) in enumerate(zip(dataloader_train[classes[0]], dataloader_train[classes[1]])):
        
        data_0 = batch_a['points'].transpose(2,1).to(device)
        data_1 = batch_b['points'].transpose(2,1).to(device)

        if data_0.shape[0] != batch_size or data_1.shape[0] != batch_size:
            continue
        
        batch_count += 1
        
        outputs = model(data_0, data_1)
        optimizer_zero_grad(optimizers)
        losses = calculate_losses(outputs, loss_params, data_0, data_1, batch_size, True)

        content_reconstruction_loss += float(losses[0])
        style_reconstruction_loss += float(losses[1])
        adversarial_loss += float(losses[2])
        chamfer_loss += float(losses[3])
        chamfer_cycle_loss += float(losses[4])
        
        optimizer_step(optimizers)

    return [content_reconstruction_loss/batch_count, style_reconstruction_loss/batch_count, adversarial_loss/batch_count, chamfer_loss/batch_count, chamfer_cycle_loss/batch_count]

def evaluate_model(model, best_results_dir, dataloader_eval, epoch, classes, batch_size):
    model.eval()
    
    batch_count = 0
    content_reconstruction_loss = 0
    style_reconstruction_loss = 0
    adversarial_loss = 0
    chamfer_loss = 0
    chamfer_cycle_loss = 0
    
    data_0, data_1 = None, None
    
    with torch.no_grad():
        for _, (batch_a, batch_b) in enumerate(zip(dataloader_eval[classes[0]], dataloader_eval[classes[1]])):

            data_0 = batch_a['points'].transpose(2,1).to(device)
            data_1 = batch_b['points'].transpose(2,1).to(device)
            eval_loss = 0

            if data_0.shape[0] != batch_size or data_1.shape[0] != batch_size:
                continue

            batch_count += 1

            outputs = model(data_0, data_1)
            losses = calculate_losses(outputs, loss_params, data_0, data_1, batch_size, False)

            content_reconstruction_loss += float(losses[0])
            style_reconstruction_loss += float(losses[1])
            adversarial_loss += float(losses[2])
            chamfer_loss += float(losses[3])
            chamfer_cycle_loss += float(losses[4])

    mean_eval_loss = (content_reconstruction_loss + style_reconstruction_loss + adversarial_loss + chamfer_loss + chamfer_cycle_loss)/batch_count

    global best_loss
    if mean_eval_loss < best_loss :
        print("Found best evaluation at epoch: " + str(epoch))
        print("\tContent loss: " + str(content_reconstruction_loss/batch_count))
        print("\tStyle loss: " + str(style_reconstruction_loss/batch_count))
        print("\tAdversarial loss: " + str(adversarial_loss/batch_count))
        print("\tChamfer loss: " + str(chamfer_loss/batch_count))
        print("\tChamfer cycle loss: " + str(chamfer_cycle_loss/batch_count))
        best_loss = mean_eval_loss
        torch.save(model, best_results_dir+"model"+str(epoch)+".pt")

def train(model, dataloaders, optimizers, device, nepoch, loss_params, batch_size, best_results_dir, classes):

    model = model.to(device)
    model.train()

    dataloader_train = dataloaders["train"]
    dataloader_eval = dataloaders["eval"]
    for epoch in range(0, nepoch):
        
        batch_losses = train_epoch(dataloader_train, model, optimizers, device, loss_params, batch_size, classes)

        print("Training Epoch " + str(epoch))
        print("\tContent loss: " + str(batch_losses[0]))
        print("\tStyle loss: " + str(batch_losses[1]))
        print("\tAdversarial loss: " + str(batch_losses[2]))
        print("\tChamfer loss: " + str(batch_losses[3]))
        print("\tChamfer cycle loss: " + str(batch_losses[4]))
        print(batch_losses)
        
#        if epoch%3 == 0:
        evaluate_model(model=model, best_results_dir=best_results_dir, epoch=epoch, dataloader_eval=dataloader_eval, classes=classes, batch_size=batch_size)


if __name__ == "__main__":
    
    opt = parser()

    dataloaders = prepare_data(dataset_class = dataset_shapenet.ShapeNet, family=opt.family, classes = [opt.class_0, opt.class_1], data_dir = opt.data_dir, batch_size = opt.batch_size)

    model = prepare_model()
    
    optimizers = prepare_optimizers(model, opt.generator_lrate, opt.discriminator_lrate)
    
    print(opt)

    loss_params = {
        "weight_chamfer" : opt.weight_chamfer,
        "weight_cycle_chamfer" : opt.weight_cycle_chamfer,
        "weight_adversarial" : opt.weight_adversarial,
        "weight_perceptual" : opt.weight_perceptual,
        "weight_content_reconstruction" : opt.weight_content_reconstruction,
        "weight_style_reconstruction" : opt.weight_style_reconstruction
    }
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Found device: ", device)

    train(model=model, dataloaders=dataloaders, optimizers=optimizers, device=device, nepoch=opt.nepoch, loss_params=loss_params, batch_size=opt.batch_size, best_results_dir=opt.best_results_dir, classes=[opt.class_0, opt.class_1])