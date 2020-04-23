from models.mesh_classifier import RegresserModel
from data.dataloader import MSDSurfTrainDataset
from util.writer import Writer
import time
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
import torch
import torch.nn as nn
from losses.loss import DiceWithCELoss
import numpy as np


# train
def train(opt):
    dataset = MSDSurfTrainDataset(opt)
    dataloader = DataLoader(dataset)
    writer = Writer(opt)

    model = RegresserModel(opt)

    seg_criterion = DiceWithCELoss()
    mesh_criterion = nn.MSELoss()

    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)

    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            img_patch = torch.from_numpy(data['img_patch']).view(-1, 1, 128, 128, 64)
            mask_patch = torch.from_numpy(data['mask_patch']).view(-1, 1, 128, 128, 64)
            vs = torch.from_numpy(data['vs'])
            meshes = data['mesh']
            edges = torch.from_numpy(data['edge_features']).float()

            img_patch = img_patch.cuda()
            mask_patch = mask_patch.cuda()
            edges = edges.cuda()
            out_mask, out_map, out_edges = model(img_patch, edges, meshes)

            seg_loss = seg_criterion(out_mask, mask_patch)
            mesh_loss = mesh_criterion(out_edges, vs)

            if total_steps % opt.print_freq == 0:
                loss = mesh_model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                mesh_model.save_network('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            mesh_model.save_network('latest')
            mesh_model.save_network(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        mesh_model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(mesh_model, epoch)

        if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch)
            writer.plot_acc(acc, epoch)

    writer.close()


if __name__ == '__main__':
    opt = TrainOptions().parse()
    train(opt)
