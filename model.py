import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, GroupNorm, AdaptiveMaxPool1d 
from easydict import EasyDict
import dataset_shapenet as dataset_shapenet

# the "MLP" block that you will use the in the PointNet and CorrNet modules you will implement
# This block is made of a linear transformation (FC layer), 
# followed by a Leaky RelU, a Group Normalization (optional, depending on enable_group_norm)
# the Group Normalization (see Wu and He, "Group Normalization", ECCV 2018) creates groups of 32 channels
def MLP(channels, enable_group_norm=True):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i]//32)
            else:
                num_groups.append(1)    
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2), GroupNorm(num_groups[i], channels[i]))
                     for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2))
                     for i in range(1, len(channels))])


# PointNet module for extracting point descriptors
# num_input_features: number of input raw per-point or per-vertex features 
# 		 			  (should be 3, since we have 3D point positions in this assignment)
# num_output_features: number of output per-point descriptors (should be 32 for this assignment)
# this module should include
# - a MLP that processes each point i into a 128-dimensional vector f_i
# - another MLP that further processes these 128-dimensional vectors into h_i (same number of dimensions)
# - a max-pooling layer that collapses all point features h_i into a global shape representaton g
# - a concat operation that concatenates (f_i, g) to create a new per-point descriptor that stores local+global information
# - a MLP that transforms this concatenated descriptor into the output 32-dimensional descriptor x_i
class PointNet(torch.nn.Module):
    def __init__(self, num_input_features=3, num_output_features=1024):
        super(PointNet, self).__init__()
        self.mlp = MLP([num_input_features, 32, 64, 128])
        self.mlp_single = MLP([128, 128])
        self.maxpool = AdaptiveMaxPool1d(1)
        self.mlp_2 = MLP([256, 128, 64])
        self.lin = Lin(64, num_output_features, bias=True)


    def forward(self, x):
        f = self.mlp(x)
        x = self.mlp_single(f)
        g = self.maxpool(x.T)
        x = torch.cat((f,(g.T).expand(f.size())),dim=1)
        x = self.mlp_2(x)
        x = self.lin(x)
        return x


class ThreeDsnet(nn.Module):
    def __init__(self):
        super(ThreeDsnet, self).__init__()

       
        self.classes = ['classa','classb']

        content_encoder_0 = content_encoder_1 = PointNet()
        self.content_encoder = {self.classes[0]: content_encoder_0,
                                self.classes[1]: content_encoder_1}
        self.content_encoder = nn.ModuleDict(self.content_encoder)

        self.style_encoder = PointNet()

        decoder_0 = decoder_1 = PointNet()
        self.decoder = {self.classes[0]: decoder_0,
                                self.classes[1]: decoder_1}
        self.decoder = nn.ModuleDict(self.decoder)

    def fuse_primitives(self, points, faces, sample=True):
        """
        Merge generated surface elements in a single one and prepare data for Chamfer

        :param points: input points for each existing patch to be merged
        Input size : batch, prim, 3, npoints
        Output size : prim, prim*npoints, 3

        :param faces: faces connecting points. Necessary for meshflow which needs mesh representations

        :return points: merged surface elements in a single one
        """
        if self.opt.decoder_type.lower() == "atlasnet":
            # import pdb; pdb.set_trace()
            points = points.transpose(2, 3).contiguous()
            points = points.view(points.size(0), -1, 3)
        elif self.opt.decoder_type.lower() == "meshflow" and sample and self.flags.train:
            meshes = Meshes(verts=points, faces=faces)
            points = sample_points_from_meshes(meshes, num_samples=points.size(1))
        return points

    def forward(self, x, content_class, style_class, train=True):
        """
        :param x: a dictionary containing an input pair.
        :param content_class: category label for the input from which to extract the content.
        :param style_class: style label for the input from which to extract the style.
        :param train: boolean, True if training.
        :return:
        """

        # Extract content from the desired image x[content_class]
        content = self.content_encoder[content_class](x[content_class])
        # Extract style from the desired image x[style_class]
        style = self.style_encoder[style_class](x[style_class])
        # Decode latent codes to output pointcloud
        out = self.decoder[style_class](content, style, train=train)
        return out

    def generator_update_forward(self, x, train=True):
        # Encode
        content_0 = self.content_encoder[self.classes[0]](x[self.classes[0]][self.type_0])
        content_1 = self.content_encoder[self.classes[1]](x[self.classes[1]][self.type_1])
        style_0_prime = self.style_encoder[self.classes[0]](x[self.classes[0]][self.type_0])
        style_1_prime = self.style_encoder[self.classes[1]](x[self.classes[1]][self.type_1])

        # Decode latent codes (within domain)
        out_00 = self.decoder[self.classes[0]](content_0, style_0_prime, train=train)
        fused_out_00 = self.fuse_primitives(out_00['points_3'], out_00['faces'], train, self.sample_points_from_mesh)
        out_11 = self.decoder[self.classes[1]](content_1, style_1_prime, train=train)
        fused_out_11 = self.fuse_primitives(out_11['points_3'], out_11['faces'], train, self.sample_points_from_mesh)
        # Decode latent codes (cross domain)
        out_01 = self.decoder[self.classes[1]](content_0, style_1_prime, train=train)
        fused_out_01 = self.fuse_primitives(out_01['points_3'], out_01['faces'], train, self.sample_points_from_mesh)
        out_10 = self.decoder[self.classes[0]](content_1, style_0_prime, train=train)
        fused_out_10 = self.fuse_primitives(out_10['points_3'], out_10['faces'], train,self.sample_points_from_mesh)

        content_01 = self.content_encoder[self.classes[1]](fused_out_01)
        style_01 = self.style_encoder[self.classes[1]](fused_out_01)
        content_10 = self.content_encoder[self.classes[0]](fused_out_10)
        style_10 = self.style_encoder[self.classes[0]](fused_out_10)

        cycle_out_010 = self.decoder[self.classes[0]](content_01, style_0_prime, train=train)
        cycle_out_101 = self.decoder[self.classes[1]](content_10, style_1_prime, train=train)

        # Classify domain membership for each style transferred pointcloud
        class_00 = self.discriminate(fused_out_00, self.classes[0])
        class_01 = self.discriminate(fused_out_01, self.classes[1])
        class_11 = self.discriminate(fused_out_11, self.classes[1])
        class_10 = self.discriminate(fused_out_10, self.classes[0])

        out_0 = {'reconstruction': out_00['points_3'],  # this is the final reconstruction
                 'reconstruction_1': None if not 'points_1' in out_00 else out_00['points_1'],
                 'reconstruction_2': None if not 'points_2' in out_00 else out_00['points_2'],
                 'faces': None if not 'faces' in out_00 else out_00['faces'],
                 'style_transfer': out_01['points_3'],
                 'cycle_reconstruction': None if not cycle else cycle_out_010['points_3'],  # this is the final reconstruction
                 'cycle_reconstruction_1': None if (not cycle or not 'points_1' in cycle_out_010) else cycle_out_010['points_1'],
                 'cycle_reconstruction_2': None if (not cycle or not 'points_2' in cycle_out_010) else cycle_out_010['points_2'],
                 'reconstruction_logits': class_00,
                 'style_transfer_logits': class_01,
                 'content_code': content_0,
                 'style_code': style_0_prime,
                 'cycle_content_code': content_01,
                 'cycle_style_code': style_10}  # cycle_style_code: E^s_0((D_0(E^c(x_1), E^s_0(x_0)))

        out_1 = {'reconstruction': out_11['points_3'], # this is the final reconstruction
                 'reconstruction_1': None if not 'points_1' in out_11 else out_11['points_1'],
                 'reconstruction_2': None if not 'points_2' in out_11 else out_11['points_2'],
                 'faces': None if not 'faces' in out_11 else out_11['faces'],
                 'style_transfer': out_10['points_3'],
                 'cycle_reconstruction': None if not cycle else cycle_out_101['points_3'],  # this is the final reconstruction
                 'cycle_reconstruction_1': None if (not cycle or not 'points_1' in cycle_out_101) else cycle_out_101['points_1'],
                 'cycle_reconstruction_2': None if (not cycle or not 'points_2' in cycle_out_101) else cycle_out_101['points_2'],
                 'reconstruction_logits': class_11,
                 'style_transfer_logits': class_10,
                 'content_code': content_1,
                 'style_code': style_1_prime,
                 'cycle_content_code': content_10,
                 'cycle_style_code': style_01}  # cycle_style_code: E^s_1((D_1(E^c(x_0), E^s_1(x_1)))

        return out_0, out_1

    def discriminator_update_forward(self, x, train=True):
        # Encode
        content_0 = self.content_encoder[self.classes[0]](x[self.classes[0]][self.type_0])
        content_1 = self.content_encoder[self.classes[1]](x[self.classes[1]][self.type_1])
        style_0_prime = self.style_encoder[self.classes[0]](x[self.classes[0]][self.type_0])
        style_1_prime = self.style_encoder[self.classes[1]](x[self.classes[1]][self.type_1])

        # Decode latent codes to output pointcloud
        out_01 = self.decoder[self.classes[1]](content_0, style_1_prime, train=train)
        fused_out_01 = self.fuse_primitives(out_01['points_3'], out_01['faces'], train, self.sample_points_from_mesh)
        # Decode latent codes to output pointcloud
        out_10 = self.decoder[self.classes[0]](content_1, style_0_prime, train=train)
        fused_out_10 = self.fuse_primitives(out_10['points_3'], out_10['faces'], train, self.sample_points_from_mesh)

        # Classify domain membership for each style transferred pointcloud
        class_0 = self.discriminate(x[self.classes[0]]['points'], self.classes[0])
        class_1 = self.discriminate(x[self.classes[1]]['points'], self.classes[1])
        class_01 = self.discriminate(fused_out_01, self.classes[1])
        class_10 = self.discriminate(fused_out_10, self.classes[0])

        out_0 = {'style_transfer': out_01['points_3'],
                 'faces': None if not 'faces' in out_01 else out_01['faces'],
                 'reconstruction_logits': class_0,
                 'style_transfer_logits': class_10}

        out_1 = {'style_transfer': out_10['points_3'],
                 'faces': None if not 'faces' in out_10 else out_10['faces'],
                 'reconstruction_logits': class_1,
                 'style_transfer_logits': class_01}

        return out_0, out_1

a = ThreeDsnet()

classes = ['armchair','straight chair,side chair']
opt = {"data_dir":"/mnt/nfs/work1/mccallum/jbshah/3dsnet/dataset/data/","normalization": "UnitBall", "SVR": True, "sample": True, "number_points": 2500, "shapenet13": True}
dataset_class = dataset_shapenet.ShapeNet
dataset_train = { classes[0]: dataset_class(EasyDict(opt), 'chair', classes[0], train=True),
        classes[1]: dataset_class(EasyDict(opt), 'chair', classes[1], train=True) }

dataloader_train = {}
dataloader_train[classes[0]] = torch.utils.data.DataLoader(
                dataset_train[classes[0]],
                batch_size=16,
                shuffle=True,
            )
dataloader_train[classes[1]] = torch.utils.data.DataLoader(
                dataset_train[classes[1]],
                batch_size=16,
                shuffle=True,
            )


def train():
    
    for _, (data_a, data_b) in enumerate(zip(dataloader_train[classes[0]], dataloader_train[classes[1]])):
        data_a = EasyDict(data_a)
        data_b = EasyDict(data_b)
        data_a.points = data_a.points.to('cuda')
        data_b.points = data_b.points.to('cuda')
        if len(data_a.points) == len(data_b.points):
            
            #gen loss
            data_a.loss, data_b.loss = 0, 0
            data_a.lpips_from_target, data_b.lpips_from_target = 0, 0
            data_a.lpips_from_source, data_b.lpips_from_source = 0, 0
            data_a.lpips_rec_from_source, data_b.lpips_rec_from_source = 0, 0

            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            generator_ops(data_a, data_b)
            discriminator_ops(data_a, data_b)

            loss = data_a.loss + data_b.loss
            log.update("loss_train_gen", loss.item()
            
            log.update("loss_train_dis", loss.item())
            loss.backward()
            
            generator_optimizer.step()  # gradient update
            print_iteration_stats(loss, 'generator')
            discriminator_optimizer.step()  # gradient update
            print_iteration_stats(loss, 'discriminator')

train()