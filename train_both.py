from models.mesh_classifier import RegresserModel
from model import generate_model
from data.dataloader import MSDSurfTrainDataset
from util.writer import Writer
import time
from torch.utils.data import DataLoader
from test import run_test
from options.train_options import TrainOptions


# train
def train(opt):
    dataset = MSDSurfTrainDataset(opt)
    dataloader = DataLoader(dataset)
    writer = Writer(opt)

    seg_model = generate_model(opt)
    mesh_model = RegresserModel(opt)

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

            mesh_model.set_input(data)
            mesh_model.optimize_parameters()

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