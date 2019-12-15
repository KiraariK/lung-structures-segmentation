from argparse import ArgumentParser
import os

import cv2
import numpy as np

MASK_FILE_EXT = '.npy'
IMG_FILE_EXT = '.png'


def main(input_folder: str, output_folder: str):
    in_path = os.path.abspath(input_folder)
    out_path = os.path.abspath(output_folder)

    file_names = sorted(os.listdir(input_folder))
    for fn in file_names:
        name, ext = os.path.splitext(fn)
        if ext != MASK_FILE_EXT:
            continue

        fn_path = os.path.join(in_path, fn)
        mask = np.load(fn_path).astype(np.uint8)
        mask_image = np.where(mask > 0, 255, 0)

        out_file_name = ''.join([name, IMG_FILE_EXT])
        out_file_path = os.path.join(out_path, out_file_name)
        cv2.imwrite(out_file_path, mask_image)


if __name__ == '__main__':
    parser = ArgumentParser(description='Converts binary mask arrays to images')

    parser.add_argument(
        '-i',
        '--input_folder',
        metavar='path/to/folder/',
        type=str,
        help='A path to a folder containing files with binary arrays in NPY format.'
    )

    parser.add_argument(
        '-o',
        '--output_folder',
        metavar='path/to/folder/',
        type=str,
        help='A path to a folder to save result images.'
    )

    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
