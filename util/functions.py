import numpy as np
import cv2


def get_edge(img, edge_level=1):
    """
    Calculate edges on the image.
    Thanks to https://qiita.com/supersaiakujin/items/494cc16836738b5394c8

    Args:
        img: numpy array in HWC format.
        edge_level:
            0: thin edge line, patchy, scratchy.
            1: thick and sometimes doubly.

    Returns:
        2D uint8 image, non-zero elements are the edge pixels.
        > img[get_edge(img) > 0] = 0 # Put black edge pixels on the image.
    """
    org_shape = img.shape

    # preprocess: blur image to suppress small noise edges
    blur_kernel = np.array([1,1,1, 1,3,1, 1,1,1]) / 11
    img = cv2.filter2D(img, 256, blur_kernel)

    # preprocess: sharpen edges
    k = -1.
    sharpningKernel8 = np.array([k, k, k, k, 9.0, k, k, k, k])
    img = cv2.filter2D(img, 256, sharpningKernel8)

    # extract edges by image pyramid
    L = 2
    tmp = img.copy()
    edges = [cv2.Canny(tmp.astype(np.uint8),100,200 )]
    for idx in range(L-1):
        tmp = cv2.pyrDown(tmp)
        edges.append(cv2.Canny(tmp.astype(np.uint8),100,200 ))

    # recover size (edges[1] is currently half of org_img)
    edge = edges[0] if edge_level == 0 else cv2.resize(edges[1], org_shape[:2])

    return edge


def simplify_by_kmeans(img, K=10, quantize_color=True):
    """
    Simplify image by using kmeans clustering colors.

    Args:
        img: Input image in HWC format.
        K: Number of clusters (= colors).
        quantize_color: resulting colors are quantized.

    Returns:
        RGB numpy image array [HWC]
  
    Notes:
        .
    """
    # dilation & erusion to connect small piecies all together
    kernel = np.ones((10,10), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # ensure shape is 3d
    reshaped_from_2d = False
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
        reshaped_from_2d = True

    # apply kmeans()
    Z = img.reshape((-1, img.shape[-1]))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    img = res.reshape((img.shape))

    # quantize colors
    if quantize_color:
        # Assign quantized grayscale colors to unique colors,
        # darker grayscale color is assigned to darker unique color.
        # This is done naturally by the sorted array `uniqs` below.
        # np.unique() returns sorted array.
        uniqs = np.unique(img.reshape(-1, img.shape[-1]), axis=0)
        N = len(uniqs)
        color_map = {(255 * i) // N: uc for i, uc in enumerate(uniqs)}
        for color_code, uc in color_map.items():
            img[np.logical_and.reduce(img == uc, axis = -1), :] = color_code

    if reshaped_from_2d:
        img = img[:, :, 0]

    return img


def abstract_image_array(img, K=10, grayscale=False, add_edge=True, quantize_color=True):
    """Make abstract version of image from array.
    This will:
        - Downgrade to K colors, resulting image will have only K colors.
        - (option) Convert to grayscale.
        - Edges extracted from original image will be overlayed on top of resulting image.
        - Quantize color value.

    Returns:
        abstract version of original image.
    """

    org_img = img.copy()

    # apply grayscale to make clustering easier
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # apply clustering
    if K is not None:
        img = simplify_by_kmeans(img, K=K, quantize_color=quantize_color)

    # emphasize edge
    if add_edge:
        edge = get_edge(org_img)
        img[edge > 0] = 255

    # convert back to color image
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)        
    
    return img


def abstract_image(img_file, K=10, grayscale=False, add_edge=True,
                   quantize_color=True, resize=None):
    """Make abstract version of image from file.

    Returns:
        orginal image: image loaded from file as is.
        abstract image: converted one.
    """

    img = cv2.imread(str(img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize is not None:
        img = cv2.resize(img, resize)
    
    return img, abstract_image_array(img, K=K, grayscale=grayscale,
                                     add_edge=add_edge, quantize_color=quantize_color)
