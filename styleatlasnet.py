"""
Copyright (c) 2021, Mattia Segu
Licensed under the MIT License (see LICENSE for details)
"""

import pymesh
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
from torch.autograd import Variable


class SquareTemplate():
    def __init__(self, device=0):
        self.device = device
        self.dim = 2
        self.npoints = 0

    def get_random_points(self, shape, device="gpu0"):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 2, x ... x]
        """
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.uniform_(0, 1)
        return Variable(rand_grid)

    def get_regular_points(self, npoints=2500, device="gpu0"):
        """
        Get regular points on a Square
        Return Tensor of Size [x, 3]
        """
        if not self.npoints == npoints:
            self.npoints = npoints
            vertices, faces = self.generate_square(np.sqrt(npoints))
            self.mesh = pymesh.form_mesh(vertices=vertices, faces=faces)  # 10k vertices
            self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0,1).contiguous().unsqueeze(0)

        return Variable(self.vertex[:, :2].contiguous().to(device))

    @staticmethod
    def generate_square(grain):
        """
        Generate a square mesh from a regular grid.
        :param grain:
        :return:
        """
        grain = int(grain)
        grain = grain - 1  # to return grain*grain points
        # generate regular grid
        faces = []
        vertices = []
        for i in range(0, int(grain + 1)):
            for j in range(0, int(grain + 1)):
                vertices.append([i / grain, j / grain, 0])

        for i in range(1, int(grain + 1)):
            for j in range(0, (int(grain + 1) - 1)):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i + 1,
                              j + (grain + 1) * (i - 1)])
        for i in range(0, (int((grain + 1)) - 1)):
            for j in range(1, int((grain + 1))):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i - 1,
                              j + (grain + 1) * (i + 1)])

        return np.array(vertices), np.array(faces)

def get_num_ada_norm_params(model):
    # return the number of AdaNorm parameters needed by the model
    num_ada_norm_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveBatchNorm1d" or m.__class__.__name__ == "AdaptiveInstanceNorm":
            num_ada_norm_params += 2 * m.norm.num_features
    return num_ada_norm_params

class AdaptiveBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm1d, self).__init__()
        self.norm = nn.BatchNorm1d(num_features, eps, momentum, affine)

    def forward(self, x, params):
        
        a = params[:, :params.size(1) // 2].unsqueeze(2)
        b = params[:, params.size(1) // 2:].unsqueeze(2)
        print(x.shape)
        print(a.shape)
        print(b.shape)
        return a*x + b * self.norm(x)  # TODO(msegu): ouch, why a * x and not just a? Must be a bug

class Mapping2Dto3D(nn.Module):
    """
    Core Atlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. Note that this is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self):
        self.style_bottleneck_size = 512
        self.bottleneck_size = 1024
        self.input_size = 2 #for square
        self.dim_output = 3
        self.hidden_neurons = 512
        self.num_layers = 2
        self.num_layers_style = 1
        # self.activation = nn.ReLU()
        self.decode_style = True

        super(Mapping2Dto3D, self).__init__()
        self.activation = nn.ReLU()

        print(
            f"New MLP decoder : hidden size {self.hidden_neurons}, num_layers {self.num_layers}, "
            f"num_layers_style {self.num_layers_style}, activation {self.activation}")
        
        self.conv1 = torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)
        self.conv_list = nn.ModuleList(
            [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])
        self.conv_list_style = nn.ModuleList(
            [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers_style)])
        self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        norm = torch.nn.BatchNorm1d 
        self.bn1 = norm(self.bottleneck_size)
        self.bn2 = norm(self.hidden_neurons)
        self.bn_list = nn.ModuleList([norm(self.hidden_neurons) for i in range(self.num_layers)])
        self.bn_list_style = nn.ModuleList(
            [norm(self.hidden_neurons) for i in range(self.num_layers_style)])


    def forward(self, x, content, style):
        x = self.conv1(x) + content
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))

        if self.decode_style:
            x = x + style
            for i in range(self.num_layers_style):
                x = self.activation(self.bn_list_style[i](self.conv_list_style[i](x)))
        return self.last_conv(x)



class AdaptiveMapping2Dto3D(nn.Module):
    """
    Core Atlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. Note that this is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self):
        self.bottleneck_size = 1024
        self.input_size = 2 #for square
        self.dim_output = 3
        self.hidden_neurons = 512
        self.num_layers = 2
        self.num_layers_style = 1
        # self.activation = nn.ReLU()
        self.decode_style = True

        super(AdaptiveMapping2Dto3D, self).__init__()

        self.activation = nn.ReLU()
        print(
            f"New MLP decoder : hidden size {self.hidden_neurons}, num_layers {self.num_layers}, "
            f"activation {self.activation}")

        self.conv1 = torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)

        self.conv_list = nn.ModuleList(
            [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])

        self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        self.bn1 = AdaptiveBatchNorm1d(self.bottleneck_size)
        self.bn2 = AdaptiveBatchNorm1d(self.hidden_neurons)

        self.bn_list = nn.ModuleList([AdaptiveBatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)])

        

    def forward(self, x, content, adabn_params):
        x = self.conv1(x) + content
        x = self.activation(self.bn1(x, adabn_params[:, 0:self.bottleneck_size * 2]))
        x = self.activation(self.bn2(
            self.conv2(x), adabn_params[:,
                           self.bottleneck_size * 2:
                           self.bottleneck_size * 2 + self.hidden_neurons * 2]))
        for i in range(self.num_layers):
            x = self.activation(self.bn_list[i](
                self.conv_list[i](x),
                adabn_params[:,
                self.bottleneck_size * 2 + (1 + i) * self.hidden_neurons * 2:
                self.bottleneck_size * 2 + (2 + i) * self.hidden_neurons * 2]))

        return self.last_conv(x)

class StyleAtlasnet(nn.Module):

    def __init__(self, number_points, nb_primitives):
        """
        Core Atlasnet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        :param opt: 
        """
        super(StyleAtlasnet, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Define number of points per primitives
        self.nb_primitives = nb_primitives
        self.nb_pts_in_primitive = number_points // nb_primitives
        self.nb_pts_in_primitive_eval = number_points // nb_primitives

        # Initialize templates
        self.template = [SquareTemplate(self.device) for i in range(0, self.nb_primitives)]

        # Initialize deformation networks
        self.decoder = nn.ModuleList([Mapping2Dto3D() for i in range(0, nb_primitives)])


    def forward(self, content_latent_vector, style_latent_vector, train=True):
        """
        Deform points from self.template using the embedding latent_vector
        :param train: a boolean indicating training mode
        :param content_latent_vector: an opt.bottleneck size vector encoding the content of a 3D shape.
                                      size : batch, bottleneck
        :param style_latent_vector: an opt.bottleneck size vector encoding the style of a 3D shape.
                                      size : batch, bottleneck
        :return: A deformed pointcloud of size : batch, nb_prim, num_point, 3
        """
        if train:
            input_points = [self.template[i].get_random_points(
                torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)),
                content_latent_vector.device) for i in range(self.nb_primitives)]
        else:
            input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval,
                                                                device=content_latent_vector.device)
                            for i in range(self.nb_primitives)]

        # Deform each patch
        num_adabn_params = get_num_ada_norm_params(self.decoder[0])
        # print(num_adabn_params)
        # print(style_latent_vector.shape)
        # print(seld.decoder[0])
        # output_patches = [self.decoder[i](input_points[i],
        #                                     content_latent_vector.unsqueeze(2),
        #                                     style_latent_vector[:, i*num_adabn_params:(i+1)*num_adabn_params]
        #                                     ).unsqueeze(1)
        #                     for i in range(0, self.nb_primitives)]
        output_patches = [self.decoder[i](input_points[i],
                                            content_latent_vector.unsqueeze(2),
                                            style_latent_vector.unsqueeze(2)
                                            ).unsqueeze(1)
                            for i in range(0, self.nb_primitives)]
                            

        output_points = torch.cat(output_patches, dim=1)

        output = {
            'faces': None,
            # 'points_1': pred_y1,
            # 'points_2': pred_y2,
            'points_3': output_points.contiguous(),  # batch, nb_prim, num_point, 3
        }
        return output
