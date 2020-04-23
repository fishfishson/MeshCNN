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
        self.seg_net = DAResNet3d(opt.nclasses, opt.seg_inplanes)
        self.down_convs = [opt.input_nc] + opt.ncf
        self.up_convs = opt.ncf[::-1] + [1]
        self.pool_res = [opt.ninput_edges] + opt.pool_res
        self.mesh_net = networks.MeshEncoderDecoder(self.pool_res,
                                                    self.down_convs,
                                                    self.up_convs,
                                                    blocks=opt.resblocks,
                                                    transfer_data=True)

    @staticmethod
    def add_feature(edges, fmaps, vs):
        size = fmaps.size()
        fmaps = fmaps.view((-1, 8) + size[1:]).permute(0, 2, 1, 3, 4, 5)  # b k 8 128 128 64
        fmaps_full = torch.zeros_like(fmaps).view(size(0), size(1), 256, 256, 128)

        fmaps_full[:, :, :128, :128, :64] = fmaps[:, :, 0]
        fmaps_full[:, :, :128, :128, 64:] = fmaps[:, :, 1]
        fmaps_full[:, :, 128:, :128, :64] = fmaps[:, :, 2]
        fmaps_full[:, :, 128:, :128, 64:] = fmaps[:, :, 3]
        fmaps_full[:, :, :128, 128:, :64] = fmaps[:, :, 4]
        fmaps_full[:, :, :128, 128:, 64:] = fmaps[:, :, 5]
        fmaps_full[:, :, 128:, 128:, :64] = fmaps[:, :, 6]
        fmaps_full[:, :, 128:, 128:, 64:] = fmaps[:, :, 7]

        n_b = size(0)
        n_k = size(1)
        n_e = edges.shape[-1]
        edges_map = torch.zeros((n_b, n_k, n_e))
        for i in range(n_b):
            fmap_full = fmaps_full[i]
            v1_ids = edges[i, :, 0]
            v2_ids = edges[i, :, 1]
            v1 = vs[i, v1_ids]
            v2 = vs[i, v2_ids]
            fmap_full_v1 = fmap_full[:, v1[:, 0], v1[:, 1], v1[:, 2]]
            fmap_full_v2 = fmap_full[:, v2[:, 0], v2[:, 1], v2[:, 2]]
            edges_map[i] = (fmap_full_v1 + fmap_full_v2) / 2

        return edges_map

    def forward(self, img_patch, edge_fs, edges, meshes, vs):
        out_mask, out_fmap = self.seg_net(img_patch)
        return out_mask, out_fmap
        # edges_map = self.add_feature(edges, out_fmap, vs)
        # edges_input = torch.cat([edge_fs, edges_map], dim=1)
        # out_edges = self.mesh_net(edges_input, meshes)


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
