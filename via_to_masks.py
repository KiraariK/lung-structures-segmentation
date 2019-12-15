from argparse import ArgumentParser
import json
import os
from typing import List

import cv2
import numpy as np


IMAGES_WIDTH = 512
IMAGES_HEIGHT = 512
POLYGON_PARENTS_NUMBER_TO_RENDER = [1, 5, 9, 13, 17, 21]
POLYGON_TO_RENDER_COLOR = 255
POLYGON_PARENTS_NUMBER_TO_CUT = [3, 7, 11, 15, 19, 23]
POLYGON_TO_CUT_COLOR = 0


def count_parents(hierarchy: np.ndarray, contour_idx: int):
    """Counts a number of parent contours for the contour.

    Args:
        hierarchy: numpy array of OpenCV contours hierarchy, which has a shape
            (num contours, 4), where the second axis represents contour properties.
        contour_idx: index of a contour in a contours list.

    Returns:
        A number of parent contours for the current contour.
    """
    counter = 0
    contour_info = hierarchy[contour_idx]

    while contour_info[3] != -1:
        counter += 1
        contour_info = hierarchy[contour_info[3]]

    return counter


def render_contour(contours: List[np.ndarray],
                   hierarchy: np.ndarray,
                   image: np.ndarray,
                   contour_idx: int = 0,
                   rendered_contours_idx: List[int] = list()):
    """Renders a contour as a polygon of a certain color.

    Polygons which have odd indices in a set of odd numbers have a white color.
    Polygons which have even indices in a set of odd numbers have a black color.

    Args:
        contours: OpenCV contours presented as a list of numpy arrays of shape
            (num points, 1, 2), where the third axis represents point coordinates.
        hierarchy: numpy array of OpenCV contours hierarchy, which has a shape
            (num contours, 4), where the second axis represents contour properties.
        image: image numpy array of shape (height, width).
        contour_idx: index of a current contour in a contours list.
        rendered_contours_idx: a list of already rendered contour indices,
            this is a mutable argument.
    """
    contour_info = hierarchy[contour_idx]
    if contour_info[3] != -1:
        if contour_idx not in rendered_contours_idx:
            num_parents = count_parents(hierarchy, contour_idx)
            if num_parents in POLYGON_PARENTS_NUMBER_TO_RENDER:
                cv2.fillPoly(image, [contours[contour_idx]], 255)
                rendered_contours_idx.append(contour_idx)
            elif num_parents in POLYGON_PARENTS_NUMBER_TO_CUT:
                cv2.fillPoly(image, [contours[contour_idx]], 0)
                rendered_contours_idx.append(contour_idx)
            else:
                pass  # Other regions should not be rendered as a polygon


def create_polygon_image(image: np.ndarray, contours: List[np.ndarray], hierarchy: np.ndarray) -> np.ndarray:
    """Creates a binary image containing region polygons.

    Only outer polygons in the result image are filled with color,
        inner polygons have a color of a background.

    Args:
        image: image on which polygons should be drawn,
            presented as a numpy array of shape (height, width).
        contours: OpenCV contours presented as a list of numpy arrays of shape
            (num points, 1, 2), where the third axis represents point coordinates.
        hierarchy: numpy array of OpenCV contours hierarchy, which has a shape
            (1, num contours, 4), where the third axis represents contour properties.

    Returns:
        A binary image containing polygons filled with color. An image presented
            as a numpy array of shape (height, width).
    """
    res_image = image.copy()
    contours_hierarchy = hierarchy[0]

    rendered_contours = []
    for i in range(len(contours)):
        render_contour(contours, contours_hierarchy, res_image, i, rendered_contours)

    return res_image


def main(annotation_file: str,
         output_folder: str,
         images_height: int = IMAGES_HEIGHT,
         images_width: int = IMAGES_WIDTH):
    """Main function.

    Args:
        annotation_file: path to a VIA annotation file ain a json format.
        output_folder: a path to a folder where the result mask images need to be saved.
        images_height: height of images for which annotations were created.
        images_width: width of images for which annotations were created.
    """
    file_path = os.path.abspath(annotation_file)
    out_folder = os.path.abspath(output_folder)
    os.makedirs(out_folder, exist_ok=True)

    with open(file_path, 'r') as file:
        annotations = json.load(file)

    for img_id in annotations:
        regions = annotations[img_id]['regions']
        clear_filename = os.path.splitext(annotations[img_id]['filename'])[0]

        annotation_contours = []
        for r in regions:
            x_points = r['shape_attributes']['all_points_x']
            y_points = r['shape_attributes']['all_points_y']

            coordinates = np.zeros((len(x_points), 2), dtype=np.int32)
            for i, (x, y) in enumerate(zip(x_points, y_points)):
                coordinates[i] = np.array([x, y], dtype=np.int32)

            annotation_contours.append(coordinates)

        if len(annotation_contours) != 0:
            # Retrieving contours as CV2 objects with hierarchy
            help_image = np.zeros((images_height, images_width), dtype=np.uint8)
            cv2.drawContours(help_image, annotation_contours, -1, (255, 255, 255), 1)
            contours, hierarchy = cv2.findContours(help_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # Draw region polygons
            mask_image = np.zeros((images_height, images_width), dtype=np.uint8)
            mask_image = create_polygon_image(mask_image, contours, hierarchy)
        else:
            mask_image = np.zeros((images_height, images_width), dtype=np.uint8)

        # Save the mask
        mask_path = os.path.join(out_folder, '.'.join([clear_filename, 'png']))
        cv2.imwrite(mask_path, mask_image)


if __name__ == '__main__':
    parser = ArgumentParser(description='Creates binary masks from VIA annotation file '
                                        'and save them on disk as PNG images.')

    parser.add_argument(
        '-af',
        '--annotation_file',
        metavar='path/to/file',
        type=str,
        help='A path to a VIA annotation file in JSON format.'
    )

    parser.add_argument(
        '-o',
        '--output_folder',
        metavar='path/to/folder/',
        type=str,
        help='A path to a folder where the result mask images need to be saved.'
    )

    parser.add_argument(
        '-ih',
        '--image_height',
        metavar='HEIGHT',
        type=int,
        help='Height of images for which annotations were '
             'created. Default is {}.'.format(IMAGES_HEIGHT),
        default=IMAGES_HEIGHT
    )

    parser.add_argument(
        '-iw',
        '--image_width',
        metavar='WIDTH',
        type=int,
        help='Width of images for which annotations were '
             'created. Default is {}.'.format(IMAGES_WIDTH),
        default=IMAGES_WIDTH
    )

    args = parser.parse_args()

    main(args.annotation_file, args.output_folder, args.image_height, args.image_width)
