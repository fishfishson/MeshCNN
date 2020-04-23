import os, glob, re
import ants
import numpy as np
from tqdm import tqdm
import mcubes
import elasticdeform
import open3d as o3d
from data.asm import align_shape, npcvtobj


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


def crop(img, size=(256, 256, 128)):
    z, y, x = img.shape
    new_img = np.zeros(size)
    ori = (z // 2 - size[0] // 2,
           y // 2 - size[1] // 2,
           size[2] // 2 - x // 2)
    new_img[:, :, ori[2]: ori[2] + x] = img[
                                        ori[0]: ori[0] + size[0],
                                        ori[1]: ori[1] + size[1],
                                        :]
    return new_img


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

        # crop
        img_np = img_.numpy()
        gt_np = gt_.numpy()
        img_np = crop(img_np)
        gt_np = crop(gt_np)

        # normal
        img_np = normalize(img_np)

        # sample surf init
        for j in tqdm(range(num_surf)):
            gt_dfm = elasticdeform.deform_random_grid(gt_np, 4, 4, 0)
            gt_dfm_smooth = mcubes.smooth_gaussian(gt_dfm, 1)
            v, e = mcubes.marching_cubes(gt_dfm_smooth, 0)
            mcubes.export_obj(v, e, os.path.join(save_dir, 'surfs_unaligned',
                                                 '{:0>2d}_{:0>2d}surf_init.obj'.format(i + 1, j + 1)))

        # write image
        img_nii = ants.from_numpy(img_np, img_.origin, img_.spacing, img_.direction, img_.has_components, img_.is_rgb)
        gt_nii = ants.from_numpy(gt_np, gt_.origin, gt_.spacing, gt_.direction, gt_.has_components, gt_.is_rgb)
        ants.image_write(img_nii, os.path.join(save_dir, 'images', '{:0>2d}img.nii'.format(i + 1)))
        ants.image_write(gt_nii, os.path.join(save_dir, 'labels', '{:0>2d}gt.nii'.format(i + 1)))

        gt_smooth = mcubes.smooth_gaussian(gt_np, 1)
        v, e = mcubes.marching_cubes(gt_smooth, 0)
        mcubes.export_obj(v, e, os.path.join(save_dir, 'surfs_unaligned', '{:0>2d}surf.obj'.format(i + 1)))


def surf_preprocess(root, task='Heart2'):
    gt_surf_list = glob.glob(os.path.join(root, task, 'surfs_unaligned', '*surf.obj'))
    gt_surf_list.sort()
    init_surf_list = glob.glob(os.path.join(root, task, 'surfs_unaligned', '*surf_init.obj'))
    init_surf_list.sort()

    temp_mesh = o3d.io.read_triangle_mesh(os.path.join(root, task, 'temp.obj'))
    temp_pts = np.asarray(temp_mesh.vertices)

    temp_aligned_surfs = align_shape(temp_pts, gt_surf_list)
    temp_aligned_init_surfs = align_shape(temp_pts, init_surf_list)
    # temp is the first one
    temp_aligned_surfs[0] = temp_pts

    for i in range(temp_aligned_surfs.shape[0]):
        temp_aligned_surf = npcvtobj(temp_mesh, temp_aligned_surfs[i])
        o3d.io.write_triangle_mesh(os.path.join(root, task, 'surfs', '{:0>2d}surf.obj'.format(i + 1)),
                                   temp_aligned_surf)
    for i in range(temp_aligned_init_surfs.shape[0]):
        temp_aligned_init_surf = npcvtobj(temp_mesh, temp_aligned_init_surfs[i])
        m = re.match(r'([0-9]*)_([0-9]*)surf_init.obj', os.path.basename(init_surf_list[i]))
        ids = m.group(1)
        num = m.group(2)
        o3d.io.write_triangle_mesh(os.path.join(root, task, 'surfs', '{}_{}surf_init.obj'.format(ids, num)),
                                   temp_aligned_init_surf)


def get_processed_list(root='', task='Heart2'):
    img_list = glob.glob(os.path.join(root, task, 'images', '*nii'))
    img_list.sort()

    gt_list = glob.glob(os.path.join(root, task, 'labels', '*nii'))
    gt_list.sort()

    gt_surf_list = glob.glob(os.path.join(root, task, 'surfs', '*surf.obj'))
    gt_surf_list.sort()

    init_surf_list = glob.glob(os.path.join(root, task, 'surfs', '*surf_init.obj'))
    init_surf_list.sort()

    surf_list = []
    for i in range(len(gt_surf_list)):
        surf_pair = dict()
        surf_pair['gt'] = gt_surf_list[i]
        surf_id = re.match(r'([0-9]*)surf.obj', os.path.basename(gt_surf_list[i])).group(1)
        corr_init_surfs = []
        for j in range(len(init_surf_list)):
            if os.path.basename(init_surf_list[j]).startswith(surf_id):
                corr_init_surfs.append(init_surf_list[j])
        surf_pair['init'] = corr_init_surfs
        surf_list.append(surf_pair)

    return img_list, gt_list, surf_list


def split_list(root, task, num_surf, proj_data_dir):
    img_list, gt_list, surf_list = get_processed_list(root, task)
    n = len(img_list)
    n_train = int(0.8 * n)

    shuffle_id = np.random.permutation(np.arange(n))
    img_array = np.array(img_list)[shuffle_id]
    gt_array = np.array(gt_list)[shuffle_id]
    surf_array = np.array(surf_list)[shuffle_id]

    train_img = img_array[:n_train]
    train_gt = gt_array[:n_train]
    train_surf = surf_array[:n_train]

    train_img = np.repeat(train_img, num_surf)
    train_gt = np.repeat(train_gt, num_surf)
    train_gt_surf = []
    train_init_surf = []
    for i in range(n_train):
        train_gt_surf_i = train_surf[i]['gt']
        train_init_surf_i = train_surf[i]['init']
        train_init_surf += train_init_surf_i
        train_gt_surf.append(train_gt_surf_i)
    train_gt_surf = np.repeat(np.array(train_gt_surf), num_surf)
    train_init_surf = np.array(train_init_surf)
    assert train_gt_surf.shape[0] == train_init_surf.shape[0]

    test_img = img_array[n_train:]
    test_gt = gt_array[n_train:]
    test_surf = surf_array[n_train:]

    test_img = np.repeat(test_img, num_surf)
    test_gt = np.repeat(test_gt, num_surf)
    test_gt_surf = []
    test_init_surf = []
    for i in range(n - n_train):
        test_gt_surf_i = test_surf[i]['gt']
        test_init_surf_i = test_surf[i]['init']
        test_init_surf += test_init_surf_i
        test_gt_surf.append(test_gt_surf_i)
    test_gt_surf = np.repeat(np.array(test_gt_surf), num_surf)
    test_init_surf = np.array(test_init_surf)

    assert test_gt_surf.shape[0] == test_init_surf.shape[0]

    train_list = np.vstack([train_img, train_gt, train_gt_surf, train_init_surf]).T
    test_list = np.vstack([test_img, test_gt, test_gt_surf, test_init_surf]).T

    np.savetxt(os.path.join(proj_data_dir, 'train_list.txt'), train_list, fmt='%s')
    np.savetxt(os.path.join(proj_data_dir, 'test_list.txt'), test_list, fmt='%s')


def main():
    root = '/home/zyuaq/mesh/data/MSD'
    proj_data_dir = '/home/zyuaq/mesh/MeshCNN/datasets/'
    task = 'Heart2'
    num_surf = 5
    # process_MSD(root, task, num_surf)
    # surf_preprocess(root, task)
    split_list(root, task, num_surf, proj_data_dir)


if __name__ == '__main__':
    main()
