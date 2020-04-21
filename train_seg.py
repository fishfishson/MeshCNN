from setting import parse_opts
from data.dataloader import MSDTrainDataset
from model import generate_model
import torch
from torch import optim
from torch.utils.data import DataLoader
import time
from util.logger import log
import os
from losses.loss import DiceLoss


def train(dataset, data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))

    print("Current setting is:")
    print(sets)
    print("\n\n")

    criterion = DiceLoss()

    if not sets.no_cuda:
        criterion = criterion.cuda()

    model.train()
    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))

        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks = batch_data

            if not sets.no_cuda:
                volumes = volumes.cuda()
                label_masks = label_masks.cuda()

            optimizer.zero_grad()
            out_masks = model(volumes)

            # calculating loss
            loss = criterion(out_masks, label_masks)
            loss.backward()
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}' \
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss.item(), avg_batch_time))

            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    # if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)

                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id))
                    torch.save({
                        'ecpoch': epoch,
                        'batch_id': batch_id,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        model_save_path)

        dataset.train_sample(sets.sample_number)

    print('Finished training')
    

if __name__ == '__main__':
    # settting
    sets = parse_opts()

    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)
    print(model)

    # optimizer
    params = [
        {'params': parameters['base_parameters'], 'lr': sets.learning_rate},
        {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}
    ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
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
    training_dataset = MSDTrainDataset(sets.train_list, [128, 128, 32], sets.phase, False)
    training_dataset.train_sample(sets.sample_number)
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers,
                             pin_memory=sets.pin_memory)

    # training
    train(training_dataset, data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs,
          save_interval=sets.save_intervals,
          save_folder=sets.save_folder, sets=sets)
