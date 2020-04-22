import os, glob
import ants
import time
import numpy as np
from tqdm import tqdm
import mcubes
import elasticdeform


def get_ACDC_list(root, phase='training'):
    gt_list = glob.glob(os.path.join(root, phase, '*/*gt.nii.gz'))
    gt_list.sort()

    img_list = glob.glob(os.path.join(root, phase, '*/*[0-9].nii.gz'))
    img_list.sort()

    return img_list, gt_list


def get_MSD_list(root, task='Heart2'):
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


def process_MSD(root, task='Heart2', num_surf=5):
    img_list, gt_list = get_MSD_list(root, task)
    save_dir = os.path.join(root, task)

    n = len(img_list)
    for i in range(n):
        print('Process img: {}'.format(img_list[i]))

        img = ants.image_read(img_list[i])
        gt = ants.image_read(gt_list[i])

        # iso-resample
        img_ = iso_resample(img, [1.5, 1.5, 1.5], islabel=False)
        gt_ = iso_resample(gt, [1.5, 1.5, 1.5], islabel=True)

        # normal
        img_ = normalize(img_)

        # sample surf init
        for j in tqdm(range(num_surf)):
            gt_np = gt_.numpy()
            gt_dfm = elasticdeform.deform_random_grid(gt_np, 4, 4, 0)
            gt_dfm_smooth = mcubes.smooth_gaussian(gt_dfm, 1)
            v, e = mcubes.marching_cubes(gt_dfm_smooth, 0)
            mcubes.export_obj(v, e, os.path.join(save_dir, 'surfs_unaligned',
                                                 '{:0>2d}_{:0>2d}_init_surf.obj'.format(i + 1, j + 1)))

        # write image
        ants.image_write(img_, os.path.join(save_dir, 'images', '{:0>2d}img.nii'.format(i + 1)))
        ants.image_write(gt_, os.path.join(save_dir, 'labels', '{:0>2d}gt.nii'.format(i + 1)))

        gt_np = gt_.numpy()
        gt_smooth = mcubes.smooth_gaussian(gt_np, 1)
        v, e = mcubes.marching_cubes(gt_smooth, 0)
        mcubes.export_obj(v, e, os.path.join(save_dir, 'surfs_unaligned', '{:0>2d}surf.obj'.format(i + 1)))


def surf_preprocess(root):
    pass


def get_processed_list(root='', task='Heart'):
    img_list = glob.glob(os.path.join(root, task, 'images', '*nii'))
    img_list.sort()

    gt_list = glob.glob(os.path.join(root, task, 'labels', '*nii'))
    gt_list.sort()

    surf_list = glob.glob(os.path.join(root, task, 'surfs', '*.obj'))
    surf_list.sort()

    return img_list, gt_list, surf_list


def split_list(root, proj_data_dir):
    img_list, gt_list, surf_list = get_processed_list(root)
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

    np.savetxt(os.path.join(proj_data_dir, 'train_list.txt'), train_list, fmt='%s')
    np.savetxt(os.path.join(proj_data_dir, 'test_list.txt'), test_list, fmt='%s')


def main():
    root = '/home/zyuaq/mesh/data/MSD'
    proj_data_dir = '/home/zyuaq/mesh/MeshCNN/datasets/'
    process_MSD(root)
    # surf_preprocess(root)
    # img_list, gt_list, surf_list = get_processed_list(root)
    # print(img_list)
    # print('*' * 10)
    # print(gt_list)
    # print('*' * 10)
    # print(surf_list)
    # split_list(root, proj_data_dir)


if __name__ == '__main__':
    main()
