from models.mesh_classifier import RegresserModel, patch
import time
from data import DataLoader
from setting import parse_opts
import torch
import torch.nn as nn
from losses.loss import DiceWithCELoss
from util.logger import log
from metrics.DiceEval import diceEval, AverageMeter
import os
from models.networks import get_scheduler


# train
def train(dataloader, model, optimizer, scheduler, total_epochs, save_interval, save_dir, opt):
    batches_per_epoch = len(dataloader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))

    print("Current setting is:")
    print(opt)
    print("\n\n")

    seg_criterion = DiceWithCELoss()
    mesh_criterion = nn.MSELoss()

    if not opt.no_cuda:
        seg_criterion = seg_criterion.cuda()
        mesh_criterion = mesh_criterion.cuda()

    train_time_sp = time.time()
    meter = AverageMeter()
    dice = diceEval(opt.nclasses)

    for epoch in range(total_epochs):

        model.train()

        log.info('Start epoch {}'.format(epoch))

        log.info('lr = {}'.format(scheduler.get_lr()))

        dice.reset()
        meter.reset()

        for batch_id, data in enumerate(dataloader):
            batch_id_sp = epoch * batches_per_epoch

            optimizer.zero_grad()

            img = torch.from_numpy(data['img']).float()
            mask = torch.from_numpy(data['mask']).long()
            img_patch = patch(img, opt.patch_size)
            mask_patch = patch(mask, opt.patch_size)
            img_patch = img_patch.unsqueeze(1)

            mesh = data['mesh']
            edges = data['edges']
            ve = data['ve']

            img_patch = img_patch.cuda()
            mask_patch = mask_patch.cuda()

            vs = data['vs'].astype('int')
            edge_fs = torch.from_numpy(data['edge_features']).float().cuda()
            gt_vs = torch.from_numpy(data['gt_vs']).float().cuda()
            vtx = torch.from_numpy(vs).float().cuda()

            out_mask, edge_offsets = model(img_patch, edge_fs, edges, vs, mesh)
            n_b = vtx.shape[0]
            n_v = vtx.shape[1]
            for i in range(n_b):
                for j in range(n_v):
                    e = ve[i, j]
                    edge_offset = edge_offsets[i, :, e]
                    offset = torch.mean(edge_offset, dim=1)
                    vtx[i, j] += offset

            seg_loss = seg_criterion(out_mask, mask_patch)
            mesh_loss = mesh_criterion(vtx, gt_vs)
            loss = 0.5 * seg_loss + mesh_loss
            loss.backward()

            optimizer.step()

            meter.update(loss.item())
            dice.addBatch(out_mask.max(1)[1], mask_patch)

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, dice = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, meter.avg, dice.getMetric(), avg_batch_time))

            if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_dir, epoch, batch_id)
                model_save_dir = os.path.dirname(model_save_path)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)

                log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
                torch.save({
                    'epoch': epoch,
                    'batch_id': batch_id,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    model_save_path)

        scheduler.step()

    print('Finished training')


if __name__ == '__main__':
    opt = parse_opts()

    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        idx = int(str_id)
        if idx >= 0:
            opt.gpu_ids.append(idx)

    torch.manual_seed(opt.seed)

    dataloader = DataLoader(opt)
    model = RegresserModel(opt)

    if not opt.no_cuda:
        if len(opt.gpu_ids) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
            net_dict = model.state_dict()
        else:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001)
    scheduler = get_scheduler(optimizer, opt)

    if opt.resume_path:
        if os.path.isfile(opt.resume_path):
            print("=> loading checkpoint '{}'".format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resume_path, checkpoint['epoch']))

    train(dataloader=dataloader,
          model=model,
          optimizer=optimizer,
          scheduler=scheduler,
          total_epochs=opt.nepoch,
          save_interval=opt.save_intervals,
          save_dir=opt.save_dir,
          opt=opt)
