import numpy as np
import cv2
import argparse
# -------- my imports --------
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from gmm import GMM
import igraph


GC_BGD = 0  # Hard bg pixel (has to be in the background in the end)
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel (at the start of the algorithm - in the foreground)


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert from absolute coordinates
    w -= x
    h -= y

    # Initialize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD

    index_row = rect[1]+rect[3]//2
    index_col = rect[0]+rect[2]//2
    # make the single pixel of the middle of the rectangle to be GC_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initialize_GMMs(img, mask)

    num_iters = 1000

    for i in range(num_iters):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initialize_gmm_of_mask_value(init_img, init_mask, mask_value, n_components):
    filtered_pixels = init_img[init_mask == mask_value]
    # normalized_pixels = filtered_pixels / 255.0
    k_means = KMeans(n_clusters=5)
    k_means.fit(filtered_pixels)

    # Replace each pixel with the centroid of its cluster
    clustered_pixels = k_means.cluster_centers_[k_means.labels_]

    # put the labels in the original image
    clustered_image = np.zeros_like(init_img)
    clustered_image[init_mask == mask_value] = clustered_pixels.astype(np.uint8)

    # cv2.imshow('Clustered Image', clustered_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gmm_list = []

    for i in range(n_components):
        current_cluster_indices = np.where(k_means.labels_ == i)[0]
        current_cluster_pixels = filtered_pixels[current_cluster_indices]
        current_cluster_mean = np.empty((current_cluster_pixels.shape[1],))
        covariance_matrix, current_cluster_mean = cv2.calcCovarMatrix(current_cluster_pixels.T, current_cluster_mean,
                                                                      flags=cv2.COVAR_NORMAL | cv2.COVAR_COLS)
        covariance_matrix = covariance_matrix / current_cluster_pixels.shape[0]
        current_component_weight = current_cluster_pixels.shape[0] / filtered_pixels.shape[0]
        current_gmm = GMM(current_cluster_mean, np.linalg.inv(covariance_matrix), np.linalg.det(covariance_matrix),
                          current_component_weight)
        gmm_list.append(current_gmm)

        # gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
        # gmm.fit(current_cluster_pixels)

    return gmm_list


def convert_custom_gmm_to_library_gmm(gmm_list):
    n_components = len(gmm_list)
    means = np.array([gmm.mean for gmm in gmm_list])
    covariance_inverses = np.array([gmm.covariance_inverse for gmm in gmm_list])
    weights = np.array([gmm.component_weight for gmm in gmm_list])

    gmm_model = GaussianMixture(n_components=n_components, covariance_type="full")
    gmm_model.weights_ = weights
    gmm_model.means_ = means
    gmm_model.precisions_cholesky_ = covariance_inverses
    return gmm_model


def initialize_GMMs(img, mask, n_components=5):
    # need to store 4 things for each gaussian model:
    # µ – the mean (an RGB triple)
    # Σ^(−1) – the inverse of the covariance matrix (a 3x3 matrix)
    # detΣ – the determinant of the covariance matrix (a real)
    # π – a component weight (a real)

    bg_gmm_list = initialize_gmm_of_mask_value(img, mask, GC_BGD, n_components)
    bgGMM = convert_custom_gmm_to_library_gmm(bg_gmm_list)

    fg_gmm_list = initialize_gmm_of_mask_value(img, mask, GC_PR_FGD, n_components)
    fgGMM = convert_custom_gmm_to_library_gmm(fg_gmm_list)

    return bgGMM, fgGMM


def update_gmm_of_pixels(pixels, gmm_model):
    # implement running the likelihood of a each gmm on each pixel
    # save the index of the one with the maximum likelihood

    pixel_likelihoods = gmm_model._estimate_weighted_log_prob(pixels)
    pixels_max_log_likelihood_indices = np.argmax(pixel_likelihoods, axis=1)

    labels, counts = np.unique(pixels_max_log_likelihood_indices, return_counts=True)

    # somehow update the gmms based on the result
    updated_gmm_list = []
    for i in range(gmm_model.n_components):
        current_cluster_pixels = pixels[pixels_max_log_likelihood_indices == i]
        if len(current_cluster_pixels) == 0:
            continue
        current_cluster_mean = np.empty((current_cluster_pixels.shape[1],))
        covariance_matrix, current_cluster_mean = cv2.calcCovarMatrix(current_cluster_pixels.T, current_cluster_mean,
                                                                      flags=cv2.COVAR_NORMAL | cv2.COVAR_COLS)
        covariance_matrix = covariance_matrix / current_cluster_pixels.shape[0]
        current_component_weight = current_cluster_pixels.shape[0] / pixels.shape[0]
        current_gmm = GMM(current_cluster_mean, np.linalg.inv(covariance_matrix), np.linalg.det(covariance_matrix),
                          current_component_weight)
        updated_gmm_list.append(current_gmm)

    return updated_gmm_list


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    new_bg_gmm_list = update_gmm_of_pixels(img[mask == GC_BGD], bgGMM)
    new_bgGMM = convert_custom_gmm_to_library_gmm(new_bg_gmm_list)

    new_fg_gmm_list = update_gmm_of_pixels(img[mask == GC_PR_FGD], fgGMM)
    new_fgGMM = convert_custom_gmm_to_library_gmm(new_fg_gmm_list)

    return new_bgGMM, new_fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]
    energy = 0

    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    correct_pixels_amount = np.count_nonzero(predicted_mask == gt_mask)
    accuracy = correct_pixels_amount / predicted_mask.size

    intersection = np.logical_and(predicted_mask, gt_mask)
    union = np.logical_or(predicted_mask, gt_mask)
    jaccard_similarity = np.sum(intersection) / np.sum(union)
    return accuracy, jaccard_similarity
    # return 100, 100


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='llama', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    # Take the required image
    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    # Take the rectangle of the required image
    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    # read the image
    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
