import cv2
import numpy as np


def rtpose_preprocess(image):
    image = image.astype(np.float32)
    image = image / 256. - 0.5
    image = image.transpose((2, 0, 1)).astype(np.float32)

    return image


def vgg_preprocess(image):
    image = image.astype(np.float32) / 255.
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = image.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
    return preprocessed_img

def factor_closest(num, factor, is_ceil=True):
    num = np.ceil(float(num) / factor) if is_ceil else np.floor(float(num) / factor)
    num = int(num) * factor
    return num


def crop_with_factor(im, dest_size=None, factor=32, is_ceil=True):
    if dest_size==None:
        dest_size=im.shape[0]
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.
    # if max_size is not None and im_size_min > max_size:
    im_scale = float(dest_size) / im_size_min
    im = cv2.resize(im, None, fx=im_scale, fy=im_scale)

    h, w, c = im.shape
    new_h = factor_closest(h, factor=factor, is_ceil=is_ceil)
    new_w = factor_closest(w, factor=factor, is_ceil=is_ceil)
    im_croped = np.zeros([new_h, new_w, c], dtype=im.dtype)
    im_croped[0:h, 0:w, :] = im

    return im_croped, im_scale, im.shape