import torch
from . import networks
import torch.nn as nn
from os.path import join
from util.util import seg_accuracy, print_network
from models.resunet import DAResNet3d
from losses.loss import DiceWithCELoss
import numpy as np


class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        #
        self.nclasses = opt.nclasses

        # load/define networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        labels = torch.from_numpy(data['label']).long()
        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])

    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

    ##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            # compute number of correct
            pred_class = out.data.max(1)[1]
            label_class = self.labels
            self.export_segmentation(pred_class.cpu())
            correct = self.get_accuracy(pred_class, label_class)
        return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])


class RegresserModel(nn.Module):
    def __init__(self, opt):
        super(RegresserModel, self).__init__()
        self.opt = opt
        self.seg_net = DAResNet3d(opt.nclasses, opt.seg_inplanes)
        self.down_convs = [opt.input_nc] + opt.ncf
        self.up_convs = opt.ncf[::-1] + [1]
        self.pool_res = [opt.ninput_edges] + opt.pool_res
        self.mesh_net = networks.MeshEncoderDecoder(self.pool_res,
                                                    self.down_convs,
                                                    self.up_convs,
                                                    blocks=opt.resblocks,
                                                    transfer_data=True)

    def add_feature(self, edges, batch_fmap, vs):
        b = self.opt.batch_size
        k = self.opt.seg_inplanes
        n_e = edges.shape[1]
        edges_map = torch.zeros((b, k, n_e)).float().cuda()
        for i in range(b):
            fmap = batch_fmap[i]
            v1_ids = edges[i, :, 0]
            v2_ids = edges[i, :, 1]
            v1 = vs[i, v1_ids]
            v2 = vs[i, v2_ids]
            fmap_v1 = fmap[:, v1[:, 0], v1[:, 1], v1[:, 2]]
            fmap_v2 = fmap[:, v2[:, 0], v2[:, 1], v2[:, 2]]
            edges_map[i] = (fmap_v1 + fmap_v2) / 2

        return edges_map

    def patch(self, img):
        ps = self.opt.patch_size
        assert len(img.shape) == 4
        patches = img.unfold(1, ps[0], ps[0]).unfold(2, ps[1], ps[1]).unfold(3, ps[2], ps[2])
        patches = patches.contiguous().view(-1, 1, ps[0], ps[1], ps[2])
        return patches

    def unpatch(self, patches):
        size = patches.size()
        channel = size[1]
        patches = patches.permute(1, 0, 2, 3, 4)
        patches = patches.view(channel, -1, 2, 2, 2, size[2], size[3], size[4])
        patches = patches.permute(0, 1, 2, 5, 3, 6, 4, 7)
        patches = patches.reshape(channel, self.opt.batch_size, 2 * size[2], 2 * size[3], 2 * size[4])
        patches = patches.permute(1, 0, 2, 3, 4)
        return patches

    def forward(self, img_patch, edge_fs, edges, vs, meshes):
        out_mask, out_fmap = self.seg_net(img_patch)
        batch_fmap = self.unpatch(out_fmap)
        edge_fmap = self.add_feature(edges, batch_fmap, vs)
        edge_input = torch.cat([edge_fs, edge_fmap], dim=1)
        edge_offset = self.mesh_net(edge_input, meshes)
        return out_mask, edge_offset


class SurfLoss(nn.Module):
    def __init__(self):
        super(SurfLoss, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, out_edges, gt_vs, vs, ve):
        n_b = out_edges.size(0)
        loss = 0
        for i in range(n_b):
            v = vs[i]
            gt_v = gt_vs[i]
            es = ve[i]
            out_edge = out_edges[i]
            n_v = v.shape[0]
            for j in range(n_v):
                e = es[j]
                e_f = out_edge[0, e]
                offset = torch.mean(e_f)
                loss += self.l2_loss(v[j] + offset, gt_v[j])
        return loss
