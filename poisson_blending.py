import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


def get_laplacian_operator_of_pixel(img, i, j, color):
    value = 4 * img[i, j, color].astype(np.int32)
    if i > 0:
        value -= img[i - 1, j, color].astype(np.int32)
    if i < img.shape[0] - 1:
        value -= img[i + 1, j, color].astype(np.int32)
    if j > 0:
        value -= img[i, j - 1, color].astype(np.int32)
    if j < img.shape[1] - 1:
        value -= img[i, j + 1, color].astype(np.int32)
    return value


def get_pixel_neighbors_indices(i, j):
    return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]


def is_pixel_edge(i, j, img_mask):
    if img_mask[i, j] == 0:
        return False
    neighbors = get_pixel_neighbors_indices(i, j)
    for neighbor in neighbors:
        if neighbor[0] < 0 or neighbor[0] >= img_mask.shape[0]:
            continue
        if neighbor[1] < 0 or neighbor[1] >= img_mask.shape[1]:
            continue
        if img_mask[neighbor] == 0:
            return True
    return False


def fill_sparse_poisson_matrix(img_mask, index_map, indices_xs, indices_ys, num_of_indices):
    sparse_matrix = scipy.sparse.lil_matrix((num_of_indices, num_of_indices), dtype=np.float32)
    # print(f"total indices: {num_of_indices}")
    for i in range(num_of_indices):
        # if i % 5000 == 0:
        #    print(f"current index: {i}")
        x, y = indices_xs[i], indices_ys[i]
        sparse_matrix[i, i] = 4
        index_neighbors = get_pixel_neighbors_indices(x, y)

        for n_x, n_y in index_neighbors:
            if (0 <= n_x < img_mask.shape[0]) and (0 <= n_y < img_mask.shape[1]) and img_mask[n_x, n_y] != 0:
                neighbors_place_in_sparse = index_map[n_x, n_y]
                sparse_matrix[i, neighbors_place_in_sparse] = -1

    return sparse_matrix


def poisson_blend(im_src, im_tgt, im_mask, center):
    src_center_x = int(im_src.shape[0] / 2)
    tgt_center_x = center[1]
    x_offset = tgt_center_x - src_center_x

    src_center_y = int(im_src.shape[1] / 2)
    tgt_center_y = center[0]
    y_offset = tgt_center_y - src_center_y

    masked_non_zero_indices = np.nonzero(im_mask)
    num_of_non_zero_masked = len(masked_non_zero_indices[0])
    masked_xs, masked_ys = masked_non_zero_indices
    index_map = -np.ones_like(im_mask, dtype=np.int32)
    index_map[masked_non_zero_indices] = np.arange(num_of_non_zero_masked)

    im_blend = np.copy(im_tgt)

    poisson_sparse_matrix = fill_sparse_poisson_matrix(im_mask, index_map, masked_xs, masked_ys, num_of_non_zero_masked)

    for color in range(im_src.shape[-1]):
        poisson_vector = np.zeros(num_of_non_zero_masked, dtype=np.float32)
        for i in range(num_of_non_zero_masked):
            x, y = masked_xs[i], masked_ys[i]
            laplacian_value = get_laplacian_operator_of_pixel(im_src, x, y, color)
            poisson_vector[i] = laplacian_value

            if is_pixel_edge(x, y, im_mask):
                index_neighbors = get_pixel_neighbors_indices(x, y)

                for neighbor in index_neighbors:
                    if neighbor[0] < 0 or neighbor[0] >= im_src.shape[0]:
                        continue
                    if neighbor[1] < 0 or neighbor[1] >= im_src.shape[1]:
                        continue
                    if im_mask[neighbor[0], neighbor[1]] == 0:
                        poisson_vector[i] += im_tgt[neighbor[0] + x_offset, neighbor[1] + y_offset, color]

        x_sol = spsolve(poisson_sparse_matrix.tocsr(), poisson_vector)
        x_sol = np.clip(x_sol, 0, 255)
        for i in range(num_of_non_zero_masked):
            x, y = masked_xs[i], masked_ys[i]
            im_blend[x + x_offset, y + y_offset, color] = x_sol[i]

    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana1.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='target file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
