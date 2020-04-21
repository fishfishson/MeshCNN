import os
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset
import copy


class MSDTrainDataset(Dataset):

    def __init__(self, train_lst, patch_size, phase, flip):
        self.coords = dict()
        with open(train_lst, 'r') as f:
            self.train_lst = [line.strip() for line in f]
        print("Processing {} datas".format(len(self.train_lst)))
        self.patch_size = patch_size
        self.flip = flip
        self.phase = phase
        self.data_lst = dict()

        for idx in range(len(self.train_lst)):
            ith_info = self.train_lst[idx].split(" ")
            img_name = os.path.join(ith_info[0])
            label_name = os.path.join(ith_info[1])
            assert os.path.isfile(img_name)
            assert os.path.isfile(label_name)
            img = nib.load(img_name)  # We have transposed the data from WHD format to DHW
            assert img is not None
            mask = nib.load(label_name)
            assert mask is not None

            # img = np.transpose(img, (2, 0, 1))
            # img = np.clip(img, -1, 1)

            if self.phase == 'train':
                # add random noise
                noise = np.random.rand(img.shape[0], img.shape[1], img.shape[2]) * 0.2 - 0.1  # [-0.1, 0.1]
                img = img + noise
                scale = np.random.rand(img.shape[0], img.shape[1], img.shape[2]) * 0.2 - 0.1 + 1  # [0.9, 1.1]
                img = img * scale

            if self.phase == 'validate':
                print('Wrong')
                exit(0)

            # if self.phase == 'validate':
            #    img = np.pad(img, ((0, self.patch_size[0]//2+1),
            #                       (0, self.patch_size[1]//2+1),
            #                       (0, self.patch_size[2]//2+1)), 'constant', constant_values=(0))

            # label = np.transpose(label, (2, 0, 1))

            self.data_lst[idx] = {}
            self.data_lst[idx]['image'] = img.get_fdata()
            self.data_lst[idx]['mask'] = mask.get_fdata()

    def train_sample(self, num_patch):
        pz, py, px = self.patch_size
        self.coords['idx'] = []
        self.coords['loc'] = []

        for i in range(len(self.train_lst)):
            print("Processing {}th data".format(i + 1))
            img_shape = self.data_lst[i]['image'].shape
            count = 0
            while count < num_patch:
                x0, y0, z0 = np.random.randint(0, (img_shape[0] - pz,
                                                   img_shape[1] - py,
                                                   img_shape[2] - px), 3)
                self.coords['idx'].append(i)
                self.coords['loc'].append([x0, y0, z0])
                count += 1

    #     def validate_sample(self, subject):
    #         PX, PY, PZ = self.patch_size
    #         self.coords = {}
    #         self.coords['subject'] = []
    #         self.coords['xyz'] = []

    #         image_shape = self.data_list[subject]['image'].shape
    #         mask_shape = self.data_list[subject]['mask'].shape

    #         for x0 in range(0, image_shape[0], PX//2):
    #             for y0 in range(0, image_shape[1], PY//2):
    #                 for z0 in range(0, image_shape[2], PZ//2):
    #                     x1 = x0 + PX
    #                     y1 = y0 + PY
    #                     z1 = z0 + PZ
    #                     if x0 < 0 or x1 > image_shape[0]:
    #                         continue
    #                     if y0 < 0 or y1 > image_shape[1]:
    #                         continue
    #                     if z0 < 0 or z1 > image_shape[2]:
    #                         continue
    #                     self.coords['subject'].append(subject)
    #                     self.coords['xyz'].append([x0, y0, z0])

    #         return mask_shape

    def __getitem__(self, idx):
        idx_i = self.coords['idx'][idx]
        loc_i = self.coords['loc'][idx]

        img = self.data_lst[idx_i]['image']
        mask = self.data_lst[idx_i]['mask']

        img_patch = copy.deepcopy(img[loc_i[0]:loc_i[0] + self.patch_size[0],
                                  loc_i[1]:loc_i[1] + self.patch_size[1],
                                  loc_i[2]:loc_i[2] + self.patch_size[2]])
        mask_patch = copy.deepcopy(mask[loc_i[0]:loc_i[0] + self.patch_size[0],
                                   loc_i[1]:loc_i[1] + self.patch_size[1],
                                   loc_i[2]:loc_i[2] + self.patch_size[2]])

        if self.flip:
            flip_x = np.random.choice(2) * 2 - 1
            flip_y = np.random.choice(2) * 2 - 1
            flip_z = np.random.choice(2) * 2 - 1
            img_patch = img_patch[::flip_z, ::flip_y, ::flip_x]
            mask_patch = mask_patch[::flip_z, ::flip_y, ::flip_x]

        img_patch = img_patch.astype('float32')
        img_patch = img_patch[np.newaxis, :, :, :]
        mask_patch = mask_patch.astype('float32')

        img_patch = torch.from_numpy(img_patch).type(torch.FloatTensor)
        mask_patch = torch.from_numpy(mask_patch).type(torch.FloatTensor)

        return img_patch, mask_patch, idx_i, loc_i

    def __len__(self):
        return len(self.coords['idx'])
