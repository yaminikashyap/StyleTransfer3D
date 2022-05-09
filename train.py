from model import ThreeDsnet
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

def calculate_losses(batch_losses, outputs, loss_params, data_0, data_1, batch_size, train=True):
    adversarial_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    reconstruction_loss = l1_distance
    chamfer_loss = chamfer_dist_no_sampling

    #if not train:
    #    print(outputs)
    #Content Encoder losses
    loss_CE0_reconstruction = loss_params["weight_content_reconstruction"] * reconstruction_loss(outputs["content_encoder_prime"][0], outputs["content_encoder_outputs"][0])

    loss_CE1_reconstruction = loss_params["weight_content_reconstruction"] * reconstruction_loss(outputs["content_encoder_prime"][1], outputs["content_encoder_outputs"][1])

    if train:
        loss_CE0_reconstruction.backward(retain_graph=True)
        loss_CE1_reconstruction.backward(retain_graph=True)

    batch_losses["content_reconstruction"].append(loss_CE0_reconstruction.detach().cpu().numpy() + loss_CE1_reconstruction.detach().cpu().numpy())

    #Style Encoder losses
    loss_style_reconstruction_0 = loss_params["weight_style_reconstruction"] * (reconstruction_loss(outputs["style_encoder_primes"][0], outputs["style_encoder_reconstructed_outputs"][0]))
    loss_style_reconstruction_1 = loss_params["weight_style_reconstruction"] * (reconstruction_loss(outputs["style_encoder_primes"][1], outputs["style_encoder_reconstructed_outputs"][1]))

    if train:
        loss_style_reconstruction_0.backward(retain_graph=True)
        loss_style_reconstruction_1.backward(retain_graph=True)

    batch_losses["style_reconstruction"].append(float(loss_style_reconstruction_0.detach().cpu().numpy()) + float(loss_style_reconstruction_1.detach().cpu().numpy()))

    #Adversarial losses
    valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
    fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

    loss_gen00 = loss_params["weight_adversarial"]*mse_loss(outputs["discriminator_outputs"][0],valid)
    loss_gen01 = loss_params["weight_adversarial"]*mse_loss(outputs["discriminator_outputs"][1],valid)
    loss_gen10 = loss_params["weight_adversarial"]*mse_loss(outputs["discriminator_outputs"][2],valid)
    loss_gen11 = loss_params["weight_adversarial"]*mse_loss(outputs["discriminator_outputs"][3],valid)

    if train:
        loss_gen00.backward(retain_graph=True)
        loss_gen01.backward(retain_graph=True)
        loss_gen10.backward(retain_graph=True)
        loss_gen11.backward(retain_graph=True)

    batch_losses["generator"].append(loss_gen00.detach().cpu().numpy() + loss_gen01.detach().cpu().numpy() + loss_gen10.detach().cpu().numpy() + loss_gen11.detach().cpu().numpy())

    loss_disc0 = loss_params["weight_adversarial"] * (adversarial_loss(outputs["discriminator_outputs"][0].detach(),fake) + adversarial_loss(outputs["discriminator_outputs"][2].detach(),fake) + adversarial_loss(outputs["discriminator_outputs"][4],valid))

    loss_disc1 = loss_params["weight_adversarial"] * (adversarial_loss(outputs["discriminator_outputs"][1].detach(),fake) + adversarial_loss(outputs["discriminator_outputs"][3].detach(),fake) + adversarial_loss(outputs["discriminator_outputs"][5],valid))

    if train:
        loss_disc0.backward(retain_graph=True)
        loss_disc1.backward(retain_graph=True)

    batch_losses["discriminator"].append(loss_disc0.detach().cpu().numpy() + loss_disc1.detach().cpu().numpy())

    #Chamfer loss - Identity
    chamfer_loss_00 = loss_params["weight_chamfer"]*chamfer_loss(data_0, outputs["reconstructed_outputs"][0])
    chamfer_loss_11 = loss_params["weight_chamfer"]*chamfer_loss(data_1, outputs["reconstructed_outputs"][1])

    if train:
        chamfer_loss_00.backward(retain_graph=True)
        chamfer_loss_11.backward(retain_graph=True)

    batch_losses["chamfer"].append(chamfer_loss_00.detach().cpu().numpy() + chamfer_loss_11.detach().cpu().numpy())

    #Chamfer loss - Cycle
    chamfer_loss_010 = loss_params["weight_cycle_chamfer"]*chamfer_loss(data_0, outputs["cycle_reconstructed_outputs"][0]["points_3"].view(batch_size, -1, 3))
    chamfer_loss_101 = loss_params["weight_cycle_chamfer"]*chamfer_loss(data_1, outputs["cycle_reconstructed_outputs"][1]["points_3"].view(batch_size, -1, 3))

    if train:
        chamfer_loss_010.backward(retain_graph=True)
        chamfer_loss_101.backward(retain_graph=True)

    batch_losses["chamfer_cycle"].append(chamfer_loss_010.detach().cpu().numpy() + chamfer_loss_101.detach().cpu().numpy())

#     return [loss_CE0_reconstruction + loss_CE1_reconstruction, loss_style_reconstruction_0 + loss_style_reconstruction_1, loss_gen00 + loss_gen01 + loss_gen10 + loss_gen11, loss_disc0 + loss_disc1, chamfer_loss_00 + chamfer_loss_11, chamfer_loss_010 + chamfer_loss_101]


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
    optimizer_se0 = torch.optim.Adam(model.style_encoder_0.parameters(), lr=generator_lrate)
    optimizer_se1 = torch.optim.Adam(model.style_encoder_1.parameters(), lr=generator_lrate)
    optimizer_de0 = torch.optim.Adam(model.decoder_0.parameters(), lr=generator_lrate)
    optimizer_de1 = torch.optim.Adam(model.decoder_1.parameters(), lr=generator_lrate)

    #discriminator optimizers
    optimizer_disc0 = torch.optim.Adam(model.discriminator_0.parameters(), lr=discriminator_lrate)
    optimizer_disc1 = torch.optim.Adam(model.discriminator_1.parameters(), lr=discriminator_lrate)

    return [optimizer_ce0, optimizer_ce1, optimizer_se0, optimizer_se1, optimizer_de0, optimizer_de1, optimizer_disc0, optimizer_disc1]

def l1_distance(inputs, targets):
    return torch.mean(torch.abs(inputs - targets))

def chamfer_dist(reconstructed_points, target_points):
    reconstructed_points = torch.transpose(reconstructed_points, 1, 2)
    chamfer_loss = 0
    number_points = 0
    i = 0
    for batch_idx in range(reconstructed_points.shape[0]):
        batch_points = int(target_points.shape[1]*0.20)
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

def chamfer_dist_no_sampling(reconstructed_points, target_points):
    return torch.mean(torch.min(torch.cdist(target_points, torch.transpose(reconstructed_points.to(device), 1, 2)), dim=1)[0]) 

def train_epoch(dataloader_train, model, optimizers, device, loss_params, batch_size, classes):
    batch_losses = {
        "content_reconstruction": [],
        "style_reconstruction": [],
        "generator": [],
        "discriminator": [],
        "chamfer": [],
        "chamfer_cycle":[]
    }

    model.train()

    data_0, data_1 = None, None
    for _, (batch_a, batch_b) in enumerate(zip(dataloader_train[classes[0]], dataloader_train[classes[1]])):

        data_0 = batch_a['points'].transpose(2,1).to(device)
        data_1 = batch_b['points'].transpose(2,1).to(device)

        #print(data_0.shape)
        #print(data_1.shape)
        if data_0.shape[0] != batch_size or data_1.shape[0] != batch_size:
            continue
        outputs = model(data_0, data_1)
        optimizer_zero_grad(optimizers)
        calculate_losses(batch_losses, outputs, loss_params, data_0, data_1, batch_size, True)
        optimizer_step(optimizers)

    return batch_losses

def evaluate_model(model, best_results_dir, dataloader_eval, epoch, classes, batch_size):
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
        for _, (batch_a, batch_b) in enumerate(zip(dataloader_eval[classes[0]], dataloader_eval[classes[1]])):

            data_0 = batch_a['points'].transpose(2,1).to(device)
            data_1 = batch_b['points'].transpose(2,1).to(device)
            eval_loss = 0

            #print("Eval epoch", epoch)

            if data_0.shape[0] != batch_size or data_1.shape[0] != batch_size:
                continue

            #print("Trying to eval")

            outputs = model(data_0, data_1)
            calculate_losses(eval_losses, outputs, loss_params, data_0, data_1, batch_size, False)

    print(eval_losses)

    mean_eval_loss = numpy.mean(numpy.array((numpy.mean(eval_losses["content_reconstruction"]), numpy.mean(eval_losses["style_reconstruction"]), numpy.mean(eval_losses["generator"]), numpy.mean(eval_losses["discriminator"]), numpy.mean(eval_losses["chamfer"]), numpy.mean(eval_losses["chamfer_cycle"]) )))
    print("Mean eval loss", mean_eval_loss)

    global best_loss
    if mean_eval_loss < best_loss :
        print("Found best evaluation at epoch: " + str(epoch))
        print("\tContent loss: " + str(numpy.mean(eval_losses["content_reconstruction"])))
        print("\tStyle loss: " + str(numpy.mean(eval_losses["style_reconstruction"])))
        print("\tGenerator loss: " + str(numpy.mean(eval_losses["generator"])))
        print("\tDiscriminator loss: " + str(numpy.mean(eval_losses["discriminator"])))
        print("\tChamfer loss: " + str(numpy.mean(eval_losses["chamfer"])))
        print("\tChamfer cycle loss: " + str(numpy.mean(eval_losses["chamfer_cycle"])))
        torch.save(model, best_results_dir+"model"+str(epoch)+".pt")
        best_loss = mean_eval_loss

def train(model, dataloaders, optimizers, device, nepoch, loss_params, batch_size, best_results_dir, classes):

    model = model.to(device)

    dataloader_train = dataloaders["train"]
    dataloader_eval = dataloaders["eval"]
    for epoch in range(0, nepoch):

        batch_losses = train_epoch(dataloader_train, model, optimizers, device, loss_params, batch_size, classes)

        print("Training Epoch " + str(epoch))
        print("\tContent loss: " + str(numpy.mean(batch_losses["content_reconstruction"])))
        print("\tStyle loss: " + str(numpy.mean(batch_losses["style_reconstruction"])))
        print("\tGenerator loss: " + str(numpy.mean(batch_losses["generator"])))
        print("\tDiscriminator loss: " + str(numpy.mean(batch_losses["discriminator"])))
        print("\tChamfer loss: " + str(numpy.mean(batch_losses["chamfer"])))
        print("\tChamfer cycle loss: " + str(numpy.mean(batch_losses["chamfer_cycle"])))
        print()

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
