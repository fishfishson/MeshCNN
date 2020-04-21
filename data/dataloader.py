import os
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset
import copy


class MSDTrainDataset(Dataset):

    def __init__(self, file_lst, patch_size, patch_number, phase, flip=False):
        with open(file_lst, 'r') as f:
            self.file_lst = [line.strip() for line in f]

        print("Processing {} datas".format(len(self.file_lst)))
        self.patch_size = patch_size
        self.patch_number = patch_number
        self.flip = flip
        self.phase = phase
        self.data_lst = dict()
        self.coords = dict()
        self.coords['idx'] = []
        self.coords['loc'] = []

        for idx in range(len(self.file_lst)):
            ith_info = self.file_lst[idx].split(" ")
            img_name = os.path.join(ith_info[0])
            label_name = os.path.join(ith_info[1])

            assert os.path.isfile(img_name)
            assert os.path.isfile(label_name)
            img = nib.load(img_name).get_fdata()  # We have transposed the data from WHD format to DHW
            assert img is not None
            mask = nib.load(label_name).get_fdata()
            assert mask is not None

            img = np.transpose(img, (2, 1, 0))
            mask = np.transpose(mask, (2, 1, 0))
            # img = np.clip(img, -1, 1)

            if self.phase == 'train':
                # add random noise
                noise = np.random.rand(img.shape[0], img.shape[1], img.shape[2]) * 0.2 - 0.1  # [-0.1, 0.1]
                img = img + noise
                scale = np.random.rand(img.shape[0], img.shape[1], img.shape[2]) * 0.2 - 0.1 + 1  # [0.9, 1.1]
                img = img * scale

            self.data_lst[idx] = {}
            self.data_lst[idx]['image'] = img
            self.data_lst[idx]['mask'] = mask

    def patch_sample(self):
        pz, py, px = self.patch_size
        self.coords['idx'] = []
        self.coords['loc'] = []

        for i in range(len(self.file_lst)):
            img_shape = self.data_lst[i]['image'].shape

            assert img_shape[0] - pz > 0
            assert img_shape[1] - py > 0
            assert img_shape[2] - px > 0

            if self.phase == 'train':
                count = 0
                while count < self.patch_number:
                    z0, y0, x0 = np.random.randint(0, (img_shape[0] - pz,
                                                       img_shape[1] - py,
                                                       img_shape[2] - px), 3)
                    self.coords['idx'].append(i)
                    self.coords['loc'].append([z0, y0, x0])
                    count += 1
            else:
                zs = np.arange(0, img_shape[0] - pz, pz)
                ys = np.arange(0, img_shape[1] - pz, py)
                xs = np.arange(0, img_shape[2] - pz, px)

                zs = np.append(zs, img_shape[0] - pz)
                ys = np.append(ys, img_shape[0] - py)
                xs = np.append(xs, img_shape[0] - px)

                for z in zs:
                    for y in ys:
                        for x in xs:
                            self.coords['idx'].append(i)
                            self.coords['loc'].append([z, y, x])

    def __getitem__(self, idx):
        idx_i = self.coords['idx'][idx]
        loc_i = self.coords['loc'][idx]

        img = self.data_lst[idx_i]['image']
        mask = self.data_lst[idx_i]['mask']

        pz, py, px = self.patch_size

        img_patch = copy.deepcopy(img[loc_i[0]:loc_i[0] + pz, loc_i[1]:loc_i[1] + py, loc_i[2]:loc_i[2] + px])
        mask_patch = copy.deepcopy(mask[loc_i[0]:loc_i[0] + pz, loc_i[1]:loc_i[1] + py, loc_i[2]:loc_i[2] + px])

        img_patch = img_patch.astype('float32')
        img_patch = img_patch[np.newaxis, :, :, :]
        mask_patch = mask_patch.astype('int')

        img_patch = torch.from_numpy(img_patch).type(torch.FloatTensor)
        mask_patch = torch.from_numpy(mask_patch).type(torch.LongTensor)

        return img_patch, mask_patch, idx_i, loc_i

    def __len__(self):
        return len(self.coords['idx'])
