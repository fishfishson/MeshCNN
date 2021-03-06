from setting import parse_opts
from data.dataloader import MSDTrainDataset
from model import generate_model
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import time
from util.logger import log
import os
from losses.loss import DiceLoss
from metrics.DiceEval import diceEval, AverageMeter


def train(dataset, data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # validation
    val_dataset = MSDTrainDataset(sets.test_list, sets.patch_size, sets.sample_number, phase='val')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=sets.batch_size,
                                shuffle=True,
                                num_workers=sets.num_workers,
                                pin_memory=sets.pin_memory)

    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))

    print("Current setting is:")
    print(sets)
    print("\n\n")

    loss1 = DiceLoss()
    loss2 = nn.CrossEntropyLoss(ignore_index=-1)

    if not sets.no_cuda:
        loss1 = loss1.cuda()
        loss2 = loss2.cuda()

    train_time_sp = time.time()
    meter = AverageMeter()
    dice = diceEval(sets.n_seg_classes)
    val_dice = diceEval(sets.n_seg_classes)

    for epoch in range(total_epochs):
        model.train()

        log.info('Start epoch {}'.format(epoch))

        log.info('lr = {}'.format(scheduler.get_lr()))

        dice.reset()
        val_dice.reset()
        meter.reset()

        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks, idx_i, loc_i = batch_data
            label_masks = label_masks.long()

            if not sets.no_cuda:
                volumes = volumes.cuda()
                label_masks = label_masks.cuda()

            optimizer.zero_grad()
            out_masks = model(volumes)

            # calculating loss and update optimizer
            loss = loss1(out_masks, label_masks) + 0.5 * loss2(out_masks, label_masks)
            loss.backward()
            optimizer.step()

            # update scheduler
            scheduler.step()

            # update meter
            meter.update(loss.item())
            dice.addBatch(out_masks.max(1)[1], label_masks)

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, dice = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, meter.avg, dice.getMetric(), avg_batch_time))

            # save model
            if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
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

        # update patches
        dataset.patch_sample()

        # eval
        model.eval()
        with torch.no_grad():
            for val_batch_id, val_batch_data in enumerate(val_dataloader):
                val_volumes, val_label_masks, val_idx_i, val_loc_i = val_batch_data
                val_label_masks = val_label_masks.long()

                if not sets.no_cuda:
                    val_volumes = val_volumes.cuda()
                    val_label_masks = val_label_masks.cuda()

                val_out_masks = model(val_volumes)
                val_dice.addBatch(val_out_masks, val_label_masks)

            log.info('Epoch {}: val dice = {:.3f}'.format(epoch, dice.getMetric()))

    print('Finished training')


if __name__ == '__main__':
    # settting
    sets = parse_opts()

    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)

    # optimizer
    # if sets.phase != 'test' and sets.pretrain_path:
    #     params = [
    #         {'params': parameters['base_parameters'], 'lr': sets.learning_rate},
    #         {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}
    #     ]
    optimizer = optim.SGD(parameters, lr=sets.learning_rate, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # train from resume
    if sets.resume_path:
        if os.path.isfile(sets.resume_path):
            print("=> loading checkpoint '{}'".format(sets.resume_path))
            checkpoint = torch.load(sets.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(sets.resume_path, checkpoint['epoch']))

    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True
    training_dataset = MSDTrainDataset(sets.train_list, sets.patch_size, sets.sample_number, sets.phase, flip=False)
    training_dataset.patch_sample()
    data_loader = DataLoader(training_dataset,
                             batch_size=sets.batch_size,
                             shuffle=True,
                             num_workers=sets.num_workers,
                             pin_memory=sets.pin_memory)

    # training
    train(training_dataset, data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs,
          save_interval=sets.save_intervals,
          save_folder=sets.save_folder, sets=sets)
