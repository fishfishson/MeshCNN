from options.test_options import TestOptions
from util.writer import Writer
from data.dataloader import MSDSurfTrainDataset
from models.mesh_classifier import RegresserModel
from torch.utils.data import DataLoader


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = MSDSurfTrainDataset(opt)
    dataloader = DataLoader(dataset)
    model = RegresserModel(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataloader):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
