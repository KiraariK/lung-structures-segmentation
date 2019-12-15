from argparse import ArgumentParser
import os

import cv2
import numpy as np


IMG_EXT = '.png'
TARGET_IMG_SHAPE = (512, 512)
INTENSITY_THRESHOLD = 128


def main(input_folder: str, output_folder: str):
    in_path = os.path.abspath(input_folder)
    out_path = os.path.abspath(output_folder)

    file_names = sorted(os.listdir(in_path))
    for fn in file_names:
        name, ext = os.path.splitext(fn)
        if ext != IMG_EXT:
            continue

        fn_path = os.path.join(in_path, fn)
        original_img = cv2.imread(fn_path, cv2.IMREAD_GRAYSCALE)

        resized_img = cv2.resize(original_img, TARGET_IMG_SHAPE, cv2.INTER_LINEAR)
        resized_img = np.where(resized_img < INTENSITY_THRESHOLD, 0, 255)

        out_img_name = ''.join([name, IMG_EXT])
        out_img_path = os.path.join(out_path, out_img_name)
        cv2.imwrite(out_img_path, resized_img)


if __name__ == '__main__':
    parser = ArgumentParser(description='Resizes images to a specific resolution.')

    parser.add_argument(
        '-i',
        '--input_folder',
        metavar='path/to/folder/',
        type=str,
        help='A path to a folder containing images.'
    )

    parser.add_argument(
        '-o',
        '--output_folder',
        metavar='path/to/folder/',
        type=str,
        help='A path to a folder to save resized images.'
    )

    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
