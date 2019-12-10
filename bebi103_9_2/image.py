import scipy
import skimage

def subtract_background(im, sigma=50):
    """Subtract background of the given image with a Gaussian blur.

    Additional arguments are passed directly on to the skimage.filters.gaussian
    function.

    :param im: input image to filter
    :type im: array-like
    :param sigma: strength of Gaussian filter, defaults to 50
    :type sigma: int, optional

    :return: image with background subtracted
    :rtype: array-like
    """
    im_bg = skimage.filters.gaussian(im, sigma=sigma)
    return im - im_bg

def zero_crossing_filter(im, threshold):
    """Returns image with 1 if there is a zero crossing and 0 otherwise.

    :param im: input image to detect zero-crossing on
    :type im: array-like
    :param threshold: minimal value of the gradient, as computed by Sobel filter,
                      at the crossing to count as a crossing
    :type threshold: float

    :return: binary image with zero-crossings
    :rtype: array-like
    """
    # Square structuring element
    selem = skimage.morphology.square(3)

    # Do max filter and min filter
    im_max = scipy.ndimage.filters.maximum_filter(im, footprint=selem)
    im_min = scipy.ndimage.filters.minimum_filter(im, footprint=selem)

    # Compute gradients using Sobel filter
    im_grad = skimage.filters.sobel(im)

    # Return edges
    return ( (  ((im >= 0) & (im_min < 0))
              | ((im <= 0) & (im_max > 0)))
            & (im_grad >= threshold) )

def laplacian_of_gaussian_segmentation(
        im, sigma=0.5, threshold=0.001, min_size=500, buffer_size=5, background=0
    ):
    """Segment the image with a Laplacian-of-Gaussian filter, followed by
    zero-crossing detection.

    IMPORTANT: This function accepts the RAW image (without Gaussian filter)
               and applies a strong Gaussian filter itself.

    :param im: input image to segment
    :type im: array-like
    :param sigma: strength of Gaussian filter, defaults to 0.5
    :type sigma: float, optional
    :param threshold: minimal value of the gradient, as computed by Sobel filter,
                      at the crossing to count as a crossing, defaults to 0.001
    :type threshold: float, optional
    :param min_size: smallest allowable object size, defaults to 500
    :type min_size: int, optional
    :param buffer_size: the width of the order examined, defaults to 5
    :type buffer_size: int, optional
    :param background: pixel value to consider as background, defaults to 0
    :type background: int, optional

    :return: (labeled image, number of labels (including background))
    :rtype: tuple
    """
    # Compute LoG
    im_LoG = scipy.ndimage.filters.gaussian_laplace(im, sigma=sigma)

    # Find zero-crossings
    im_edge = zero_crossing_filter(im_LoG, threshold=threshold)

    # Skeletonize edges
    im_edge = skimage.morphology.skeletonize(im_edge)

    # Fill holes
    im_bw = scipy.ndimage.morphology.binary_fill_holes(im_edge)

    # Remove small objectes that are not bacteria
    im_bw = skimage.morphology.remove_small_objects(im_bw, min_size=min_size)

    # Clear border with large buffer size b/c LoG procedure came off border
    im_bw = skimage.segmentation.clear_border(im_bw, buffer_size=buffer_size)

    # Label binary image; backward kwarg says value in im_bw to consider backgr.
    return skimage.measure.label(im_bw, background=background, return_num=True)
