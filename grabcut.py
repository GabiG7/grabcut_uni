import numpy as np
import cv2
import argparse
# -------- my imports --------
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from gmm import GMM
from igraph import Graph
import time


GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

# Global variables
G_EDGES = []
G_WEIGHTS = []
global OLD_ENERGY
global MAX_VERTEX_N_WEIGHTS


def print_image(img, mask):
    new_mask = np.copy(mask)
    new_mask[new_mask == GC_PR_BGD] = GC_BGD
    new_mask[new_mask == GC_PR_FGD] = GC_FGD
    new_mask = cv2.threshold(new_mask, 0, 1, cv2.THRESH_BINARY)[1]
    # Apply the final mask to the input image and display the results
    img_cut = img * (new_mask[:, :, np.newaxis])
    cv2.imshow('GrabCut Mask', 255 * new_mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_beta(image):
    # Precompute shifts in all four directions needed for the calculation
    # To avoid border issues, use np.pad to extend the array
    padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')

    # Central part of the image without the borders
    center = padded_image[1:-1, 1:-1]

    # Differences from the central part to its four neighbors (up, left, up-left diagonal, up-right diagonal)
    diff_up = center - padded_image[:-2, 1:-1]  # from above
    diff_left = center - padded_image[1:-1, :-2]  # from left
    diff_up_left = center - padded_image[:-2, :-2]  # from upper left diagonal
    diff_up_right = center - padded_image[:-2, 2:]  # from upper right diagonal

    # Calculate the squared differences and sum them
    sum_of_diffs = (diff_up ** 2 + diff_left ** 2 + diff_up_left ** 2 + diff_up_right ** 2).sum()

    # Count total elements considered (same as beta_elements in the original)
    total_elements = 4 * np.prod(center.shape[:2])  # 4 times because each pixel contributes 4 times

    # Compute beta as per original formula
    beta = 1 / (2 * sum_of_diffs / total_elements)

    return beta



# Translation from each vertex index to pixel location in img (x,y) for identification
def vid_to_img_coordinates(single_row_size, vid):
    x = vid // single_row_size
    y = np.mod(vid, single_row_size)
    return x, y


# Translation from location in img (x,y) to pixel vertex index for identification
def pixel_coords_to_vid(single_row_size, x, y):
    return (x * single_row_size) + y


# Returns Nlink edge weight (single edge - {n,m})
def compute_n_link(image, x, y, n_x, n_y, beta):
    # Assume neighboring pixels, hence distance is 1 for direct neighbors (horizontal/vertical)
    euclidean_dist = 1  # Default distance for direct neighbors

    # For diagonal neighbors, the distance would be sqrt(2), which can be handled if necessary:
    if abs(x - n_x) == 1 and abs(y - n_y) == 1:
        euclidean_dist = np.sqrt(2)

    # Compute squared color distance efficiently
    color_diff = image[x, y] - image[n_x, n_y]
    color_diff_squared = np.dot(color_diff, color_diff)

    # Compute weight based on the color difference and fixed Euclidean distance
    weight = (50 / euclidean_dist) * np.exp((-beta) * color_diff_squared)

    return weight


def create_N_links(image, beta):
    # get image dimensions
    num_of_rows = image.shape[0]
    num_of_cols = image.shape[1]

    weights_matrix = np.zeros(image.shape[:2], dtype=np.float32)

    # N-Links loop
    for x in range(num_of_rows):
        for y in range(num_of_cols):

            # pixel from below
            # special case - last row
            if x < num_of_rows - 1:
                n_x = x + 1
                n_y = y
                G_EDGES.append((pixel_coords_to_vid(num_of_cols, x, y), pixel_coords_to_vid(num_of_cols, n_x, n_y)))
                current_weight = compute_n_link(image, x, y, n_x, n_y, beta)
                G_WEIGHTS.append(current_weight)
                weights_matrix[x, y] += current_weight
                weights_matrix[n_x, n_y] += current_weight

            # pixel from right
            # special case - last column
            if y < num_of_cols - 1:
                n_x = x
                n_y = y + 1
                G_EDGES.append((pixel_coords_to_vid(num_of_cols, x, y), pixel_coords_to_vid(num_of_cols, n_x, n_y)))
                current_weight = compute_n_link(image, x, y, n_x, n_y, beta)
                G_WEIGHTS.append(current_weight)
                weights_matrix[x, y] += current_weight
                weights_matrix[n_x, n_y] += current_weight

            # pixel from down left diagonal
            # special case - last row or first column
            if (x < num_of_rows - 1) and y > 0:
                n_x = x + 1
                n_y = y - 1
                G_EDGES.append((pixel_coords_to_vid(num_of_cols, x, y), pixel_coords_to_vid(num_of_cols, n_x, n_y)))
                current_weight = compute_n_link(image, x, y, n_x, n_y, beta)
                G_WEIGHTS.append(current_weight)
                weights_matrix[x, y] += current_weight
                weights_matrix[n_x, n_y] += current_weight

            # pixel from down right diagonal
            # special case - last row or last column
            if (x < num_of_rows - 1) and (y < num_of_cols - 1):
                n_x = x + 1
                n_y = y + 1
                G_EDGES.append((pixel_coords_to_vid(num_of_cols, x, y), pixel_coords_to_vid(num_of_cols, n_x, n_y)))
                current_weight = compute_n_link(image, x, y, n_x, n_y, beta)
                G_WEIGHTS.append(current_weight)
                weights_matrix[x, y] += current_weight
                weights_matrix[n_x, n_y] += current_weight

    return weights_matrix


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
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initialize_GMMs(img, mask)

    # Our addition starts here #########################################################################################
    # beta = 0.5
    # beta = 8
    beta = calculate_beta(img)
    print(f"the value of beta is: {beta}")
    weights_matrix = create_N_links(img, beta)
    global MAX_VERTEX_N_WEIGHTS
    MAX_VERTEX_N_WEIGHTS = np.max(weights_matrix)
    # Our addition ends here ###########################################################################################

    # num_iters = 1000
    num_iters = 4
    global OLD_ENERGY
    OLD_ENERGY = -1
    for i in range(num_iters):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        print(f"############ Iteration {i} - Energy value: {energy} ############")
        if check_convergence(energy):
            break

        OLD_ENERGY = energy

        #print_image(img, mask)

    # Return the final mask and the GMMs
    mask[mask == GC_PR_BGD] = GC_BGD
    mask[mask == GC_PR_FGD] = GC_FGD
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


# Define helper functions for the GrabCut algorithm
def update_gmm_of_pixels(pixels, gmm_model):
    # implement running the likelihood of a each gmm on each pixel
    # save the index of the one with the maximum likelihood

    pixel_likelihoods = gmm_model._estimate_weighted_log_prob(pixels)
    pixels_max_log_likelihood_indices = np.argmax(pixel_likelihoods, axis=1)

    labels, counts = np.unique(pixels_max_log_likelihood_indices, return_counts=True)

    # somehow update the gmm based on the result
    updated_gmm_list = []
    for i in range(gmm_model.n_components):
        current_cluster_pixels = pixels[pixels_max_log_likelihood_indices == i]
        if len(current_cluster_pixels) == 0:
            continue
        current_cluster_mean = np.empty((current_cluster_pixels.shape[1],))
        covariance_matrix, current_cluster_mean = cv2.calcCovarMatrix(current_cluster_pixels.T, current_cluster_mean,
                                                                      flags=cv2.COVAR_NORMAL | cv2.COVAR_COLS)
        covariance_matrix = covariance_matrix / current_cluster_pixels.shape[0]
        epsilon = 1e-6
        covariance_matrix += epsilon * np.eye(covariance_matrix.shape[0])

        current_component_weight = current_cluster_pixels.shape[0] / pixels.shape[0]
        current_gmm = GMM(current_cluster_mean, np.linalg.inv(covariance_matrix), np.linalg.det(covariance_matrix),
                          current_component_weight)
        updated_gmm_list.append(current_gmm)

    return updated_gmm_list


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):

    bg_sure_pixels = img[mask == GC_BGD]
    bg_probable_pixels = img[mask == GC_PR_BGD]
    bg_all_pixels = np.concatenate((bg_sure_pixels, bg_probable_pixels))
    new_bg_gmm_list = update_gmm_of_pixels(bg_all_pixels, bgGMM)
    new_bgGMM = convert_custom_gmm_to_library_gmm(new_bg_gmm_list)

    fg_sure_pixels = img[mask == GC_FGD]
    fg_probable_pixels = img[mask == GC_PR_FGD]
    fg_all_pixels = np.concatenate((fg_sure_pixels, fg_probable_pixels))
    new_fg_gmm_list = update_gmm_of_pixels(fg_all_pixels, fgGMM)
    new_fgGMM = convert_custom_gmm_to_library_gmm(new_fg_gmm_list)

    return new_bgGMM, new_fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    num_of_rows, num_of_cols = img.shape[:2]
    num_pixels = num_of_rows * num_of_cols
    fg_sink = num_pixels
    bg_source = num_pixels + 1

    # Initialize the graph with vertices and edge lists prepared for bulk addition
    graph = Graph(directed=True)
    graph.add_vertices(num_pixels + 2)  # includes source and sink

    # Prepare edges and weights in batch to minimize overhead
    edges = []
    capacities = []

    # Add N-link edges and weights from pre-stored global variables
    edges.extend(G_EDGES)
    capacities.extend(G_WEIGHTS)

    # Calculate energies (reshape operations should be outside of loops for efficiency)
    fg_energy = -fgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])
    bg_energy = -bgGMM.score_samples(img.reshape((-1, img.shape[-1]))).reshape(img.shape[:-1])

    # Loop to establish T-links
    for i in range(num_of_rows):
        for j in range(num_of_cols):
            pixel_index = i * num_of_cols + j
            if mask[i, j] == GC_BGD:
                edges.append((pixel_index, fg_sink))
                capacities.append(bg_energy[i, j])
                edges.append((bg_source, pixel_index))
                capacities.append(MAX_VERTEX_N_WEIGHTS)
            elif mask[i, j] == GC_FGD:
                edges.append((pixel_index, fg_sink))
                capacities.append(MAX_VERTEX_N_WEIGHTS)
                edges.append((bg_source, pixel_index))
                capacities.append(fg_energy[i, j])
            else:  # For soft assignments, use probabilistic energies
                edges.append((pixel_index, fg_sink))
                capacities.append(bg_energy[i, j])
                edges.append((bg_source, pixel_index))
                capacities.append(fg_energy[i, j])

    # Add all edges and capacities at once
    graph.add_edges(edges)
    graph.es['capacity'] = capacities

    # Calculate min-cut
    min_cut_result = graph.mincut(bg_source, fg_sink, capacity='capacity')
    fg_vertices, bg_vertices = min_cut_result.partition

    if fg_sink not in fg_vertices:
        fg_vertices, bg_vertices = bg_vertices, fg_vertices

    # Return partitions and the cut value (energy of the cut)
    return [fg_vertices, bg_vertices], min_cut_result.value


def update_mask(mincut_sets, mask):
    single_row_size = mask.shape[1]

    fg_vertices = mincut_sets[0]
    bg_vertices = mincut_sets[1]

    fg_sink = mask.shape[0] * mask.shape[1]
    bg_source = mask.shape[0] * mask.shape[1] + 1

    # Precompute the mask indices to avoid recalculating them in the loop
    bg_indices = [vid_to_img_coordinates(single_row_size, vertex_index) for vertex_index in bg_vertices if vertex_index != bg_source]
    fg_indices = [vid_to_img_coordinates(single_row_size, vertex_index) for vertex_index in fg_vertices if vertex_index != fg_sink]

    # Use numpy's advanced indexing to set values in one operation
    bg_indices = tuple(np.array(bg_indices).T)  # Transpose list of indices to tuple for indexing
    fg_indices = tuple(np.array(fg_indices).T)

    mask[bg_indices] = GC_PR_BGD
    mask[fg_indices] = GC_PR_FGD

    return mask


def check_convergence(energy):
    global OLD_ENERGY
    if OLD_ENERGY != -1 and np.abs(energy - OLD_ENERGY) < 1e-3:
        return True
    return False


def cal_metric(predicted_mask, gt_mask):
    correct_pixels_amount = np.count_nonzero(predicted_mask == gt_mask)
    accuracy = correct_pixels_amount / predicted_mask.size

    intersection = np.logical_and(predicted_mask, gt_mask)
    union = np.logical_or(predicted_mask, gt_mask)
    jaccard_similarity = np.sum(intersection) / np.sum(union)

    return accuracy, jaccard_similarity


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)
    sol = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
    sol = cv2.threshold(sol, 0, 1, cv2.THRESH_BINARY)[1]

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    old_mask = mask
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')
    print("--- %s seconds ---" % (time.time() - start_time))

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
