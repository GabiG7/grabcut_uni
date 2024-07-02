import numpy as np
import cv2
import argparse
# -------- my imports --------
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from gmm import GMM
from igraph import Graph

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

# Global variables
global g
global g_edges
global g_weights

# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert from absolute cordinates
    w -= x
    h -= y

    # Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initialize_GMMs(img, mask)

    # Our addition starts here #########################################################################################
    global g
    g = generate_graph(img, bgGMM, fgGMM)

    # Shows V and E of graph
    #print(g.get_vertex_dataframe().values)
    #(g.get_edge_dataframe().values)

    # Our addition ends here ###########################################################################################

    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
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


# Define helper functions for the GrabCut algorithm
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


# Translation from each vertex index to pixel location in img (x,y) for identification
def calc_location(rowsize, n):
    x = np.mod(n, rowsize)
    y = n//rowsize
    return x, y


# After graph generation - shorter version (translation from graph object) same result #### very slow compared to calc_location so maybe remove later
def location_of(n):
    global g
    return g.vs["location"][n]


# Translation from location in img (x,y) to pixel vertex index for identification
def index_of(rowsize, x, y):
    #return g.vs.find(location=(x, y)).index #<-- does the same thing
    return (rowsize*y) + x


# Returns Nlink edge weight (single edge - {n,m})
def get_Nlink(img, n, m):
    # Parameters for calculations according to the formula
    w = img.shape[1]

    n_x, n_y = calc_location(w, n)
    m_x, m_y = calc_location(w, m)

    print("NM", n, m)
    nRGB = tuple(img[n_y][n_x])
    mRGB = tuple(img[m_y][m_x])

    # Pixels euclidean distance
    dist = np.linalg.norm((n_x - m_x, n_y - m_y))

    # Pixels squared color distance
    sqr_diff_norm = np.linalg.norm(tuple(int(a) - int(b) for a, b in zip(nRGB, mRGB)))**2

    # Get beta value
    beta = get_beta(sqr_diff_norm)

    return (50/dist) * np.exp((-beta)*sqr_diff_norm)


def add_Nlinks(img, N_edges):
    N_weights = [get_Nlink(img, t[0], t[1]) for t in N_edges]
    g_edges.extend(N_edges)
    g_weights.extend(N_weights)
    return


# Returns Tlink edges weights (all edges)
def get_Tlinks(img, bgGMM, fgGMM):
    return -fgGMM.score_samples(img.reshape(-1, img.shape[2])), -bgGMM.score_samples(img.reshape(-1, img.shape[2]))


"""
NOT DONE - FORMULA ?
"""


def get_beta(sqr_norm_z):
    #return 1/(2*sqr_norm_z)
    return 1


def generate_graph(img, bgGMM, fgGMM):
    global g
    g = Graph()

    # Calculate graph size and create Graph
    h, w, v = img.shape
    num_pixels = h * w

    # Adding vertices for each pixel + 2 representatives for bgd and fgd likelihoods
    g.add_vertices(num_pixels + 2)

    # Translation from each vertex index to pixel location in img
    locations_list = [calc_location(w, i) for i in range(num_pixels)]
    locations_list.extend(["FGD source", "BGD sink"])
    g.vs["location"] = locations_list

    global g_edges
    global g_weights
    g_edges = []
    g_weights = []

    # N-Links loop
    for y in range(h):
        for x in range(w):
            # Current vertex index
            curr = index_of(w, x, y)

            curr_edges = []

            # Add neighbor edges (N-Links) by vertices indexes
            right_neighbor = index_of(w, x + 1, y)
            down_neighbor = index_of(w, x, y + 1)
            right_diag_neighbor = index_of(w, x + 1, y + 1)
            left_diag_neighbor = index_of(w, x - 1, y + 1)

            # Last row
            if y == h - 1:
                # Last pixel - No edges to add
                if x == w - 1:
                    print("LAST PIXEL")
                    continue

                # Add only right
                curr_edges = [(curr, right_neighbor)]
                add_Nlinks(img, curr_edges)
                continue

            # Last column
            if np.mod(curr + 1, w) == 0:
                # Add only down and left down diag
                curr_edges = [(curr, down_neighbor), (curr, left_diag_neighbor)]
                add_Nlinks(img, curr_edges)
                continue

            # Add only N-links neighbor edges (N-Links)
            curr_edges = [(curr, right_neighbor),
                          (curr, down_neighbor),
                          (curr, right_diag_neighbor),
                          (curr, left_diag_neighbor)]

            # First column - remove left diagonal edge
            if x == 0:
                curr_edges = curr_edges[:-1]

            # Add only N-links weights of current edges
            add_Nlinks(img, curr_edges)

    # The representatives for bgd and fgd likelihoods vertex indexes - last ones in graph: sink (background)and source (foreground)
    source = num_pixels
    sink = num_pixels + 1
    fgd_edges = []
    bgd_edges = []

    # T-Links loop
    for y in range(h):
        for x in range(w):
            # Current vertex index
            curr = index_of(w, x, y)

            # Add relevant edges - NOTE THE ORDER, SOURCE THEN SINK
            bgd_edges.extend([(curr, source)])
            fgd_edges.extend([(curr, sink)])

    # Get likelihood values for all vertices in graph - FGD and BGD
    fgd_likelihoods, bgd_likelihoods = get_Tlinks(img, bgGMM, fgGMM)

    # Add all T-links likelihoods edges (bgd and fgd - according to the required order of values from score samples)
    fgd_edges.extend(bgd_edges)
    g_edges.extend(bgd_edges)
    g_weights.extend(list(fgd_likelihoods + list(bgd_likelihoods)))

    # Add full list of edges to graph
    g.add_edges(g_edges)

    # Add energy weights at the same order of values to match - NOT SURE IF NECESSARY
    g.es["energy"] = g_weights
    print("W", len(g_weights))
    print("E", len(g_edges))

    return g


# END OF HELPER FUNCTIONS ##################################

def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]
    energy = 0

    h, w, v = img.shape
    num_pixels = h * w
    sink = num_pixels
    source = num_pixels + 1

    # NOT WORKINGGGGGGGGGGGGGGGGGGGGGGG
    # Get mincut
    global g
    min_cut_result = g.mincut(source, sink, capacity=g.es["energy"])
    print(min_cut_result)

    min_cut[0] = min_cut_result.partition[0]
    min_cut[1] = min_cut_result.partition[1]

    energy = min_cut_result.value

    # Print results
    print("Mincut value (ENERGY):", energy)
    print("Foreground partition size:", len(min_cut[0]))
    print("Background partition size:", len(min_cut[1]))

    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


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
    #cv2.imshow('Original Image', img)
    #cv2.imshow('GrabCut Mask', 255 * mask)
    #cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
