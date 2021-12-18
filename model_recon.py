import jittor
from skimage.measure import marching_cubes_lewiner
import jittor as jt
from jt import nn
from jt import init
import data.data as data
from data.data import Tree


###############################################################################
# Encoder
###############################################################################

class Sampler(nn.Module):

    def __init__(self, in_feat_size, hidden_size, probabilistic=True):
        super(Sampler, self).__init__()
        self.probabilistic = probabilistic
        self.mlp1 = nn.Linear(in_feat_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, in_feat_size)
        self.mlp2var = nn.Linear(hidden_size, in_feat_size)
        init.gauss_(self.mlp1.weight, mean=0, std=0.02)
        init.constant_(self.mlp1.bias, 0)
        init.gauss_(self.mlp2mu.weight, mean=1e-5, std=0.02)
        init.constant_(self.mlp2mu.bias, 0)
        init.gauss_(self.mlp2var.weight, mean=1e-5, std=0.02)
        init.constant_(self.mlp2var.bias, 0)
        self.leaky_relu = nn.LeakyReLU(0.02)

    def execute(self, in_feat):
        encode = self.leaky_relu(self.mlp1(in_feat))
        mu = self.mlp2mu(encode)
        if self.probabilistic:
            logvar = self.mlp2var(encode)
            std = logvar.mul(0.5).exp_()
            eps = jt.randn_like(std)
            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            return jt.concat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu

class VoxelEncoder(nn.Module): 

    def __init__(self, feat_len, hidden_size=32):
        super(VoxelEncoder, self).__init__()
        self.conv1 = nn.Conv3d(1, hidden_size, 4, padding=1, stride=1, bias=False)
        self.in1 = nn.InstanceNorm3d(hidden_size)
        self.conv2 = nn.Conv3d(hidden_size, hidden_size * 2, 4, padding=1, stride=2, bias=False)
        self.in2 = nn.InstanceNorm3d(hidden_size * 2)
        self.conv3 = nn.Conv3d(hidden_size * 2, hidden_size * 4, 4, padding=1, stride=2, bias=False)
        self.in3 = nn.InstanceNorm3d(hidden_size * 4)
        self.conv4 = nn.Conv3d(hidden_size * 4, hidden_size * 8, 4, padding=1, stride=2, bias=False)
        self.in4 = nn.InstanceNorm3d(hidden_size * 8)
        self.conv5 = nn.Conv3d(hidden_size * 8, feat_len, 3, padding=0, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.02)
        init.xavier_uniform_(self.conv1.weight)
        init.xavier_uniform_(self.conv2.weight)
        init.xavier_uniform_(self.conv3.weight)
        init.xavier_uniform_(self.conv3.weight)
        init.constant_(self.conv5.bias, 0)

    def execute(self, x):
        # batch_size = x.shape[0]
        x = x.reshape(-1, 1, 32, 32, 32)
        x = self.leaky_relu(self.in1(self.conv1(x)))
        x = self.leaky_relu(self.in2(self.conv2(x)))
        x = self.leaky_relu(self.in3(self.conv3(x)))
        x = self.leaky_relu(self.in4(self.conv4(x)))
        x = self.conv5(x).reshape(1, -1)
        return x


class PartEncoder(nn.Module): 

    def __init__(self, feat_len, latent_size, probabilistic=False):
        super(PartEncoder, self).__init__()
        self.vox_enc = VoxelEncoder(feat_len)
        self.mlp1 = nn.Linear(feat_len + 3, latent_size)
        init.gauss_(self.mlp1.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp1.bias, 0)
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.sampler = PartFeatSampler(latent_size) if probabilistic else None

    def execute(self, x, norms):
        # batch_size = x.shape[0]
        # print(x.shape)
        feat = self.leaky_relu(self.vox_enc(x))
        x = self.mlp1(jt.concat([feat, norms], -1))
        if self.sampler is not None:
            x = self.sampler(x)
        return x

class NodeEncoder(nn.Module):

    def __init__(self, latent_size):
        super(NodeEncoder, self).__init__()
        self.mlp1 = nn.Linear((latent_size + 8) * 8, latent_size * 16)
        self.mlp2 = nn.Linear(latent_size * 16, latent_size * 8)
        self.mlp3 = nn.Linear(latent_size * 8, latent_size * 2)
        self.mlp4 = nn.Linear(latent_size * 2, latent_size)
        init.gauss_(self.mlp1.weight, mean=0, std=0.02)
        init.constant_(self.mlp1.bias, 0)
        init.gauss_(self.mlp2.weight, mean=0, std=0.02)
        init.constant_(self.mlp2.bias, 0)
        init.gauss_(self.mlp3.weight, mean=0, std=0.02)
        init.constant_(self.mlp3.bias, 0)
        init.gauss_(self.mlp4.weight, mean=1e-5, std=0.02)
        init.constant_(self.mlp4.bias, 0)
        self.leaky_relu = nn.LeakyReLU(0.02)

    def execute(self, x):
        x = self.leaky_relu(self.mlp1(x))
        x = self.leaky_relu(self.mlp2(x))
        x = self.leaky_relu(self.mlp3(x))
        x = self.leaky_relu(self.mlp4(x))
        return x





class RecursiveEncoder(nn.Module):
    # batch_size=1
    def __init__(self, conf, variational=True, probabilistic=False):
        super(RecursiveEncoder, self).__init__()
        self.max_depth = conf.max_depth
        self.geo_feat_size = conf.geo_feat_size
        self.latent_size = conf.latent_size
        self.node_encoder = NodeEncoder(
            latent_size=conf.latent_size
        )

        self.part_encoder = PartEncoder(
            feat_len=conf.geo_feat_size,
            latent_size=conf.geo_feat_size
        )

        if variational:
            self.sample_encoder = Sampler(
                in_feat_size=conf.latent_size,
                hidden_size=conf.sample_hidden_size,
                probabilistic=probabilistic
            )
        else:
            self.sample_encoder = None

    def encode_node(self, node):
        geo_feat = self.part_encoder(node.voxels, node.normals)
        node.geo_feat = geo_feat
        if node.isleaf:
            padding = geo_feat.new_zeros((1, self.latent_size - self.geo_feat_size))
            return jt.concat([geo_feat, padding], -1)
        else:
            one_hot = jt.eye(8)
            child_feats = []
            for i in range(8):
                if (node.type[i] == 1):
                    child_feats.append(jt.concat([self.encode_node(node.children[i]), one_hot[i:i+1]], -1))
                else:
                    # child_feats.append(jt.concat([geo_feat.new_zeros((1, self.latent_size)), one_hot[i:i+1]], -1))
                    child_feats.append(geo_feat.new_zeros((1, self.latent_size + 8)))
            child_feats = jt.concat(child_feats).reshape(1, -1)
            return self.node_encoder(child_feats)

    def encode_structure(self, obj):
        z = self.encode_node(obj.root)
        # cell = obj.get_root_cell()
        # z = self.obj_encoder(cell[0:3], cell[3:6], node_latent)
        if self.sample_encoder is not None:
            z = self.sample_encoder(z)
        return z


###############################################################################
# Decoder
###############################################################################


class SampleDecoder(nn.Module):

    def __init__(self, latent_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(latent_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, latent_size)
        init.gauss_(self.mlp1.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp1.bias, 0)
        init.gauss_(self.mlp2.weight, mean=1e-5, std=0.02)
        init.constant_(self.mlp2.bias, 0)
        self.leaky_relu = nn.LeakyReLU(0.02)

    def execute(self, in_feat):
        x = self.leaky_relu(self.mlp1(in_feat))
        out_feat = self.leaky_relu(self.mlp2(x))
        return out_feat


class BBoxDecoder(nn.Module):

    def __init__(self, latent_size):
        super(BBoxDecoder, self).__init__()
        self.mlp1 = nn.Linear(latent_size, 3)
        self.mlp2 = nn.Linear(latent_size, 3)
        self.leaky_relu = nn.LeakyReLU(0.02)

    def execute(self, in_feat):
        center = self.mlp1(in_feat).reshape(-1)
        scale = self.leaky_relu(self.mlp2(in_feat)).reshape(-1)
        return center, scale



class IM_Tiny(nn.Module):
    def __init__(self, feat_len, hidden_size=32):
        super(IM_Tiny, self).__init__()
        self.mlp1 = nn.Linear(feat_len + 3, hidden_size * 8)
        self.mlp2 = nn.Linear(hidden_size * 8, hidden_size * 8)
        self.mlp3 = nn.Linear(hidden_size * 8, hidden_size * 8)
        self.mlp4 = nn.Linear(hidden_size * 8, hidden_size * 4)
        self.mlp5 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.mlp6 = nn.Linear(hidden_size * 2, hidden_size)
        self.mlp7 = nn.Linear(hidden_size, 1)
        init.gauss_(self.mlp1.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp1.bias, 0)
        init.gauss_(self.mlp2.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp2.bias, 0)
        init.gauss_(self.mlp3.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp3.bias, 0)
        init.gauss_(self.mlp4.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp4.bias, 0)
        init.gauss_(self.mlp5.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp5.bias, 0)
        init.gauss_(self.mlp6.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp6.bias, 0)
        init.gauss_(self.mlp7.weight, mean=1e-5, std=0.02)
        init.constant_(self.mlp7.bias, 0.5)
        self.leaky_relu = nn.LeakyReLU(0.02)
        self.sigmoid = nn.Sigmoid()

    def execute(self, net):
        x = self.leaky_relu(self.mlp1(net))
        x = self.leaky_relu(self.mlp2(x))
        x = self.leaky_relu(self.mlp3(x))
        x = self.leaky_relu(self.mlp4(x))
        x = self.leaky_relu(self.mlp5(x))
        x = self.leaky_relu(self.mlp6(x))
        pred = self.sigmoid(self.mlp7(x))

        return pred



class LeafDecoder(nn.Module):

    def __init__(self, latent_size, geo_feat_size):
        super(LeafDecoder, self).__init__()
        self.mlp1 = nn.Linear(latent_size, latent_size) # for geo
        self.mlp2 = nn.Linear(latent_size, geo_feat_size) # for geo
        self.leaky_relu = nn.LeakyReLU(0.02)
        init.gauss_(self.mlp1.weight, mean=0, std=0.02)
        init.constant_(self.mlp1.bias, 0)
        init.gauss_(self.mlp2.weight, mean=0, std=0.02)
        init.constant_(self.mlp2.bias, 0)
        # self.sigmoid = nn.Sigmoid()

    def execute(self, in_feat):
        x = self.leaky_relu(self.mlp1(in_feat))
        geo_feat = self.mlp2(x)
        return geo_feat



class NodeDecoder(nn.Module):

    def __init__(self, latent_size, geo_feat_size):
        super(NodeDecoder, self).__init__()
        self.mlp1 = nn.Linear(latent_size, latent_size * 2) # for geo
        self.mlp2 = nn.Linear(latent_size * 2, latent_size * 2) # for geo
        self.mlp3 = nn.Linear(latent_size * 2, geo_feat_size) # for geo
        self.mlp4 = nn.Linear(latent_size, latent_size * 2)# for child
        self.mlp5 = nn.Linear(latent_size * 2, latent_size * 4)# for child
        self.mlp6 = nn.Linear(latent_size * 4, latent_size * 8)# for child
        self.leaky_relu = nn.LeakyReLU(0.02)
        init.gauss_(self.mlp1.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp1.bias, 0)
        init.gauss_(self.mlp2.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp2.bias, 0)
        init.gauss_(self.mlp3.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp3.bias, 0)
        init.gauss_(self.mlp4.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp4.bias, 0)
        init.gauss_(self.mlp5.weight, mean=0.0, std=0.02)
        init.constant_(self.mlp5.bias, 0)
        init.gauss_(self.mlp6.weight, mean=1e-5, std=0.02)
        init.constant_(self.mlp6.bias, 0)
        # self.sigmoid = nn.Sigmoid()

    def execute(self, in_feat):
        x = self.leaky_relu(self.mlp1(in_feat))
        x = self.leaky_relu(self.mlp2(x))
        geo_feat = self.mlp3(x)
        x = self.leaky_relu(self.mlp4(in_feat))
        x = self.leaky_relu(self.mlp5(x))
        child_feats = self.mlp6(x).reshape(8, -1)
        # # x_t = self.leaky_relu(self.mlp3(in_feat))
        # x_t = in_feat
        # node_type = self.sigmoid(self.mlp4(x_t))
        return geo_feat, child_feats




class NodeClassifier(nn.Module):
    def __init__(self, feat_len):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(feat_len, 8)
        self.sigmoid = nn.Sigmoid()
        
    def execute(self, x):
        # x = self.leaky_relu(self.mlp1(x))
        x = self.sigmoid(self.mlp1(x))
        return x




class LeafClassifier(nn.Module):
    def __init__(self, feat_len):
        super(LeafClassifier, self).__init__()
        self.mlp1 = nn.Linear(feat_len, 1)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        x = self.sigmoid(self.mlp1(x))
        return x




class PartDecoder(nn.Module):
    def __init__(self, feat_len, hidden_size=128):
        super(PartDecoder, self).__init__()
        # self.classifier = NodeClassifier(feat_len)
        self.predictor = IM_Tiny(feat_len)
        
    def execute(self, x, in_feat):
        batch_size, num_points, _ = x.shape
        feat = in_feat.reshape(batch_size, 1, -1).expand(-1, num_points, -1)
        query = jt.concat([feat, x], -1).reshape(batch_size * num_points, -1)
        pred = self.predictor(query)
        return pred




class RecursiveDecoder(nn.Module):

    def __init__(self, conf):
        super(RecursiveDecoder, self).__init__()
        self.max_depth = conf.max_depth
        self.mc_res = conf.mc_res
        self.overlap = conf.overlap

        self.sample_decoder = SampleDecoder(
            latent_size=conf.latent_size,
            hidden_size=conf.sample_hidden_size
        )
        self.box_decoder = BBoxDecoder(
            latent_size=conf.latent_size
        )
        self.leaf_decoder = LeafDecoder(
            latent_size=conf.latent_size,
            geo_feat_size=conf.geo_feat_size
        )
        self.node_decoder = NodeDecoder(
            latent_size=conf.latent_size,
            geo_feat_size=conf.geo_feat_size
        )
        self.node_classifier = NodeClassifier(
            feat_len = conf.latent_size
        )
        self.isleaf = LeafClassifier(
            feat_len=conf.latent_size
        )
        
        self.part_decoder = PartDecoder(
            feat_len=conf.geo_feat_size
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def structure_recon_loss(self, z, gt_tree):
        losses = dict()
        root_latent = self.sample_decoder(z)
        center, scale = self.box_decoder(root_latent)
        gt_cell = gt_tree.get_root_cell()
        losses['center'] = self.mse_loss(center, gt_cell[0:3]).mean()
        losses['scale'] = self.mse_loss(scale, gt_cell[3:6]).mean()
        node_losses = self.node_recon_loss(
            node_latent=root_latent,
            gt_node=gt_tree.root
        )
        losses.update(node_losses)
        return losses

    def node_recon_loss(self, node_latent, gt_node):
        losses = dict()
        isleaf = self.isleaf(node_latent)[0]
        losses['leaf'] = self.bce_loss(isleaf, gt_node.isleaf)
        
        if gt_node.isleaf:
            geo_feat = self.leaf_decoder(node_latent)
            pred = self.part_decoder(gt_node.points.unsqueeze(0), geo_feat)
            losses['type'] = 0
            losses['latent'] = self.mse_loss(geo_feat, gt_node.geo_feat).mean()
            losses['geo'] = self.bce_loss(pred, gt_node.values).mean()
        else:
            geo_feat, child_feats = self.node_decoder(node_latent)
            node_type = self.node_classifier(node_latent)
            pred = self.part_decoder(gt_node.points.unsqueeze(0), geo_feat)
            losses['type'] = self.bce_loss(node_type, gt_node.type).sum()
            losses['latent'] = self.mse_loss(geo_feat, gt_node.geo_feat).mean()
            losses['geo'] = self.bce_loss(pred, gt_node.values).mean()
            for i in range(8):
                if (gt_node.type[i] == 1):
                    child_losses = self.node_recon_loss(
                        node_latent=child_feats[i],
                        gt_node=gt_node.children[i]
                    )
                    losses['type'] += child_losses['type']
                    losses['latent'] += child_losses['latent']
                    losses['geo'] += child_losses['geo']
        return losses


    def decode_latent(self, z, tree, center, scale):
        root_latent = self.sample_decoder(z)
        center, scale = self.box_decoder(root_latent)
        # gt_cell = gt_tree.get_root_cell()
        self.node_recon(
            node_latent=root_latent,
            node=tree.root,
            center=center,
            scale=scale
        )

    def node_recon(self, node_latent, node, center, scale, depth=0):
        if depth==3:
            geo_feat = self.leaf_decoder(node_latent)
            node.isleaf = True
            node.type = jt.zeros(8)
            node.cell = jt.concat([center, scale]).reshape(-1)
            # pred = self.part_decoder(gt_node.points.unsqueeze(0), geo_feat)
        else:
            K = jt.float32([[-1,-1,-1],
                [-1,-1,1],
                [-1,1,-1],
                [-1,1,1],
                [1,-1,-1],
                [1,-1,1],
                [1,1,-1],
                [1,1,1]])
            geo_feat, child_feats = self.node_decoder(node_latent)
            node_type = self.node_classifier(node_latent)
            node.type = (node_type>0.5)
            node.children = [Tree.Node(node) for i in range(8)]
            node.cell = jt.concat([center, scale]).reshape(-1)
            for i in range(8):
                if (node.type[i] == 1):
                    self.node_recon(
                        node_latent=child_feats[i],
                        node=node.children[i],
                        depth=depth+1,
                        center=center+K[i]*scale/4,
                        scale=scale/2
                    )
            node.isleaf = False
        node.pred_geo_feat = geo_feat

    def decode_structure(self, z, model_name='untitled'):
        tree = Tree(self.max_depth, self.overlap, model_name)
        with jt.no_grad():
            root_latent = self.sample_decoder(z)
            # center, scale = self.box_decoder(z)
            # tree.root.cell = jt.concat([center, scale]).reshape(-1)
        _ = self.decode_node(
            node_latent=root_latent, 
            node=tree.root,
            depth=0
        )
        return tree

    def decode_node(self, node_latent, node, depth):
        with jt.no_grad():
            geo_feat = self.node_decoder(node_latent)
            # l = jt.linspace(-1, 1, self.mc_res)
            # grid_points = jt.stack(jt.meshgrid(l, l, l).reshape(3,-1)).t()
            # volume = self.part_decoder(geo_feat, grid_points)
            # node.volume = volume.reshape((self.mc_res,)*3)
            node.geo_feat = geo_feat
            children_latent = self.children_decoder(geo_feat)
            children_latent = children_latent.reshape(8, 256)
            node.type = nn.Sigmoid()(self.type_mlp(children_latent)).reshape(-1)
            node.children = [Tree.Node(node) for i in range(8)]
            if depth < self.max_depth:
                for i, exists in enumerate(node.type):
                    if (exists > 0.5):
                        # node.children[i].cell = node.get_cell(i)
                        _ = self.decode_node(
                            node_latent=children_latent[i],
                            node=node.children[i],
                            depth=depth + 1
                        )
            else:
                node.type[:] = 0
            if node.type.sum() > 0:
                node.isleaf = False
        return node
