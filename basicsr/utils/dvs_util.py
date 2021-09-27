import random

import cv2


def dvs_padding(imgs, gt_size):
    h, w, _ = imgs[0].shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return imgs

    for i in range(len(imgs)):
        imgs[i] = cv2.copyMakeBorder(imgs[i], 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)

    return imgs


def dvs_paired_random_crop(imgs, gt_patch_size, scale, gt_path):
    """Paired random crop.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """
    assert scale == 1, 'only support same size now'
    for i in range(len(imgs)):
        if not isinstance(imgs[i], list):
            imgs[i] = [imgs[i]]
            assert imgs[i][0].shape == imgs[0][0].shape, 'only support same size now'

    h_lq, w_lq, _ = imgs[0][0].shape
    h_gt, w_gt = h_lq, w_lq

    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    imgs = [
        [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in im]
        for im in imgs
    ]

    if len(imgs[0]) == 1:
        imgs = [im[0] for im in imgs]

    return imgs
