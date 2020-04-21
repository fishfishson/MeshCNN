import os, glob
import ants
import time
import numpy as np
from tqdm import tqdm
import mcubes


def get_ACDC_list(root, phase='training'):
    gt_list = glob.glob(os.path.join(root, phase, '*/*gt.nii.gz'))
    gt_list.sort()

    img_list = glob.glob(os.path.join(root, phase, '*/*[0-9].nii.gz'))
    img_list.sort()

    return img_list, gt_list


def get_MSD_list(root, task='Heart'):
    img_list = glob.glob(os.path.join(root, task, 'imagesTr', '*.nii.gz'))
    img_list.sort()

    gt_list = glob.glob(os.path.join(root, task, 'labelsTr', '*.nii.gz'))
    gt_list.sort()

    return img_list, gt_list


def iso_resample(img, spacing, islabel=False):
    if islabel:
        img_re = ants.resample_image(img, spacing, False, 1)
    else:
        img_re = ants.resample_image(img, spacing, False, 0)

    return img_re


def normalize(img):
    img_norm = (img - img.mean()) / img.std()
    return img_norm


# def center_crop(img, size):
#     assert img.shape[0] >= size[0]
#     assert img.shape[1] >= size[1]
#     assert img.shape[2] >= size[2]

#     offset_x = (img.shape[0] - size[0]) // 2
#     offset_y = (img.shape[1] - size[1]) // 2
#     offset_z = (img.shape[2] - size[2]) // 2

#     img_ = img[offset_x:offset_x + size[0], 
#               offset_x:offset_y + size[1],
#               offset_x:offset_z + size[2]]

#     return img_

def process_MSD_pipe(root, task='Heart', save=None):
    img_list, gt_list = get_MSD_list(root, task)
    save_dir = os.path.join(root, task)

    n = len(img_list)
    for i in tqdm(range(n)):
        time.sleep(0.1)

        img = ants.image_read(img_list[i])
        gt = ants.image_read(gt_list[i])

        # iso-resample
        img_ = iso_resample(img, [2, 2, 2], islabel=False)
        gt_ = iso_resample(gt, [2, 2, 2], islabel=True)

        # normalize
        img_ = normalize(img_)

        # save
        ants.image_write(img_, os.path.join(save_dir, 'images', '{:0>2d}img.nii'.format(i + 1)))
        ants.image_write(gt_, os.path.join(save_dir, 'labels', '{:0>2d}gt.nii'.format(i + 1)))

        # extract surf
        gt_smooth = mcubes.smooth_gaussian(gt_.numpy(), 1)
        v, e = mcubes.marching_cubes(gt_smooth, 0)
        mcubes.export_obj(v, e, os.path.join(save_dir, 'surfs', '{:0>2d}surf.obj'.format(i + 1)))


def get_processed_list(root='data', task='Heart'):
    img_list = glob.glob(os.path.join(root, task, 'images', '*nii'))
    img_list.sort()

    gt_list = glob.glob(os.path.join(root, task, 'labels', '*nii'))
    gt_list.sort()

    surf_list = glob.glob(os.path.join(root, task, 'surfs', '*.obj'))
    surf_list.sort()

    return img_list, gt_list, surf_list


def pipe(root):
    img_list, gt_list, surf_list = get_processed_list(root)
    print(img_list)
    n = len(img_list)
    n_train = int(0.8 * n)

    shuffle_id = np.random.permutation(np.arange(n))
    img_array = np.array(img_list)[shuffle_id]
    gt_array = np.array(gt_list)[shuffle_id]
    surf_array = np.array(surf_list)[shuffle_id]

    train_img = img_array[:n_train]
    train_gt = gt_array[:n_train]
    train_surf = surf_array[:n_train]

    test_img = img_array[n_train:]
    test_gt = gt_array[n_train:]
    test_surf = surf_array[n_train:]

    train_list = np.vstack([train_img, train_gt, train_surf]).T
    test_list = np.vstack([test_img, test_gt, test_surf]).T

    np.savetxt('/home/zyuaq/mesh/MeshCNN/datasets/train_list.txt', train_list, fmt='%s')
    np.savetxt('/home/zyuaq/mesh/MeshCNN/datasets/test_list.txt', test_list, fmt='%s')


pipe('/home/zyuaq/mesh/data/MSD')
