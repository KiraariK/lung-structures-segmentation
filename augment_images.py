from argparse import ArgumentParser
import os
import shutil

import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


def main(images_folder: str, masks_folder: str, iterations: int, output_folder: str = None):
    """Main function.

    Args:
        images_folder: a path to the folder containing images.
        masks_folder: a path to the folder containing polygon masks corresponding to images.
        iterations: a number of iterations to augment the data.
        output_folder: a path to the folder where the results need to be saved.
            In this folder, two folders will be created:
            folder for result images (name retrieved from input images folder),
            folder for result masks (name retrieved from input masks folder).
    """
    images_path = os.path.abspath(images_folder)
    images_folder_name = os.path.basename(images_path)
    masks_path = os.path.abspath(masks_folder)
    masks_folder_name = os.path.basename(masks_path)

    output_path = os.path.abspath(output_folder)\
        if output_folder is not None \
        else os.path.abspath(os.path.dirname(__file__))
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    output_img_path = os.path.join(output_path, images_folder_name)
    output_msk_path = os.path.join(output_path, masks_folder_name)
    os.makedirs(output_img_path, exist_ok=True)
    os.makedirs(output_msk_path, exist_ok=True)

    # Define augmentation method (configured fo CT images).
    seq = iaa.Sequential([
        iaa.Fliplr(0.3),
        iaa.Flipud(0.15),
        iaa.Sometimes(0.5, [
            iaa.GaussianBlur(sigma=(0.0, 0.5))
        ]),
        iaa.Sometimes(0.5, [
            iaa.Affine(translate_percent={'x': (-0.05, 0.05), 'y': (-0.02, 0.02)})
        ]),
        iaa.Sometimes(0.5, [
            iaa.Affine(rotate=(-7, 7))
        ])
    ], random_order=True)

    # Load images and corresponding masks
    img_batch = None
    msk_list = []
    img_names = sorted(os.listdir(images_path))
    msk_names = sorted(os.listdir(masks_path))
    assert len(img_names) == len(msk_names), 'Image and mask counts must be equal.'

    for img_name, msk_name in zip(img_names, msk_names):
        img_path = os.path.abspath(os.path.join(images_path, img_name))
        msk_path = os.path.abspath(os.path.join(masks_path, msk_name))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        img_expanded = np.expand_dims(img, axis=0)
        if img_batch is None:
            img_batch = img_expanded
        else:
            img_batch = np.concatenate((img_batch, img_expanded), axis=0)

        msk_3d = np.expand_dims(msk, axis=2)
        msk_3d = (msk_3d / 255).astype(np.float32)
        msk_list.append(ia.SegmentationMapOnImage(msk_3d, img.shape, nb_classes=2))

    # Augment images and masks
    for iter_idx in range(iterations):
        seg_det = seq.to_deterministic()
        images_aug = seg_det.augment_images(img_batch)
        segments_aug = seg_det.augment_segmentation_maps(msk_list)

        # Transform masks into numpy array images with polygons
        masks_aug = None
        for segment in segments_aug:
            arr = segment.get_arr_int()
            arr = (arr * 255).astype(np.uint8)

            arr_expanded = np.expand_dims(arr, axis=0)
            if masks_aug is None:
                masks_aug = arr_expanded
            else:
                masks_aug = np.concatenate((masks_aug, arr_expanded), axis=0)

        # Save augmented images
        for i in range(images_aug.shape[0]):
            aug_img = images_aug[i]
            aug_msk = masks_aug[i]

            img_name = os.path.splitext(img_names[i])[0] + \
                       '_{:05d}'.format(iter_idx) + \
                       os.path.splitext(img_names[i])[1]
            msk_name = os.path.splitext(msk_names[i])[0] + \
                       '_{:05d}'.format(iter_idx) + \
                       os.path.splitext(msk_names[i])[1]
            res_img_path = os.path.join(output_img_path, img_name)
            res_msk_path = os.path.join(output_msk_path, msk_name)
            cv2.imwrite(res_img_path, aug_img)
            cv2.imwrite(res_msk_path, aug_msk)


if __name__ == '__main__':
    parser = ArgumentParser(description='Test augmentation types on the set of grayscale images')

    parser.add_argument(
        '-i',
        '--images_folder',
        metavar='path/to/folder',
        type=str,
        help='A path to the folder containing grayscale images.'
    )

    parser.add_argument(
        '-m',
        '--masks_folder',
        metavar='path/to/folder',
        type=str,
        help='A path to the folder containing mask images.'
    )

    parser.add_argument(
        '-it',
        '--iterations',
        metavar='ITERATIONS',
        type=int,
        help='A number of iterations to perform data augmentation '
             'on the full set of input images.',
        default=5
    )

    parser.add_argument(
        '-o',
        '--output_folder',
        metavar='path/to/folder',
        type=str,
        help='A path to the folder where the results needs to be saved. '
             'By default, it is a folder where this script is stored.',
        default=None
    )

    args = parser.parse_args()

    main(args.images_folder, args.masks_folder, args.iterations, args.output_folder)
