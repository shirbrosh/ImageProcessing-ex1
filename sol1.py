import numpy as np
from imageio import imread
import skimage.color as skimage
from matplotlib import pyplot as plt

GRAY_SCALE = 1
NORMALIZED = 255
RGB_DIM = 3
CONVERT_TO_YIQ = np.array(
    [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    """
    This function reads an image file and converts it into a given representation.
    :param filename: the filename of an image on disk(could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the
        output should be a grayscale image(1) or an RGB image(2).
    :return: An image, represented by a matrix of type np.float64 with intensities.
    """
    image = imread(filename)

    # checks if the image is already from type float64
    if not isinstance(image, np.float64):
        image.astype(np.float64)
        image = image / NORMALIZED

    # checks if the output image should be grayscale
    if representation == GRAY_SCALE:
        image = skimage.rgb2gray(image)
    return image


def imdisplay(filename, representation):
    """
    This function utilizes the read_image function to display an image in a given
    representation.
    :param filename: the filename of an image on disk(could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the
        output should be a grayscale image(1) or an RGB image(2).
    """
    if representation == GRAY_SCALE:
        plt.imshow(read_image(filename, representation), cmap=plt.cm.gray)
    else:
        plt.imshow(read_image(filename, representation))
    plt.show()


def rgb2yiq(imRGB):
    """
    This function transforms an RGB image into the YIQ color space.
    :param imRGB: RGB image to transform, height*width*3 np.float64 matrix in the
        [0,1] range
    :return: a YIQ image
    """
    imYIQ = np.empty(imRGB.shape)
    for i in range(3):
        imYIQ[:, :, i] = (CONVERT_TO_YIQ[i][0] * imRGB[:, :, 0]) + (
                CONVERT_TO_YIQ[i][1] * imRGB[:, :, 1]) + (
                                 CONVERT_TO_YIQ[i][2] * imRGB[:, :, 2])
    return imYIQ


def yiq2rgb(imYIQ):
    """
    This function is the inverse for the rgb2yiq function -transforms an YIQ image
    into the RGB color space.
    :param imYIQ: YIQ image to transform, height*width*3 np.float64 matrix in the
        [0,1] range
    :return: an RGB image
    """
    imRGB = np.empty(imYIQ.shape)
    inv_CONVERT_TO_YIQ = np.linalg.inv(CONVERT_TO_YIQ)
    for i in range(3):
        imRGB[:, :, i] = (inv_CONVERT_TO_YIQ[i][0] * imYIQ[:, :, 0]) + (
                inv_CONVERT_TO_YIQ[i][1] * imYIQ[:, :, 1]) + (
                                 inv_CONVERT_TO_YIQ[i][2] * imYIQ[:, :, 2])
    return imRGB


def histogram_equalization_algorithm(image):
    """
    This function operates the histogram equalization algorithm on the given image
    :param image: the image on which the histogram equalization algorithm will run
    :return: a list [im_eq, hist_orig, hist_eq] where:
            im_eq - the equalized Y channel of the origin image.
            hist_origin- is a 256 bin histogram of the original image
            hist_eq- is a 256 bin histogram of the equalized image
    """
    hist_orig, bounds = np.histogram(image, bins=np.arange(257))
    lut = calculate_eq(hist_orig)
    im_eq = lut[image.astype(np.uint8)]
    hist_eq, bounds = np.histogram(im_eq, bins=np.arange(257))
    im_eq_float = (im_eq / NORMALIZED).astype(np.float64)
    return [im_eq_float, hist_orig, hist_eq]


def calculate_eq(hist_orig):
    """
    This function calculates the parameters required for the equalization process
    :param hist_orig: is a 256 bin histogram of the original image
    :return: lut- a matching look up table for the given histogram
    """
    cumulative_hist = np.cumsum(hist_orig)
    s_m = np.nonzero(hist_orig)[0][0]
    s_k = cumulative_hist[-1]
    stretched_k = (NORMALIZED * (cumulative_hist[np.arange(256)] - s_m)) / (
            s_k - s_m)
    lut = np.round(stretched_k)
    return lut


def histogram_equalize(im_orig):
    """
    This function performs histogram equalization of a given grayscale or RGB image
    :param im_orig: the input grayscale or RGB float64 image with values on [0,1]
    :return: a list [im_eq, hist_orig, hist_eq] where:
            im_eq - the equalized image. grayscale or RGB  float64 image with values
                [0,1]
            hist_origin- is a 256 bin histogram of the original image
            hist_eq- is a 256 bin histogram of the equalized image
    """
    # if the image is RGB:
    if len(im_orig.shape) == RGB_DIM:
        imYIQ = rgb2yiq(im_orig)
        imYIQ_Y = imYIQ[:, :, 0] * NORMALIZED
        result_list = histogram_equalization_algorithm(imYIQ_Y)
        imYIQ[:, :, 0] = result_list[0]
        result_list[0] = yiq2rgb(imYIQ)
        result_list[0] = np.clip(result_list[0], 0, 1)
        return result_list

    # if the image is grayscale:
    else:
        result_list = histogram_equalization_algorithm(im_orig * NORMALIZED)
        result_list[0] = np.clip(result_list[0], 0, 1)
        return result_list


def z_initialization(hist, n_quant):
    """
    This function returns the initial division such that each segment will contains
    approximately the same amount of pixels
    :param hist: the original histogram to divide into segments
    :param n_quant: the amount of segments to divide
    :return: array of the segment indexes
    """
    cumsum = np.cumsum(hist)
    threshold_value = int(cumsum[-1] / n_quant)
    z_i_arr = np.zeros(n_quant + 1).astype(int)
    z_i_arr[0] = 0
    cur_threshold_value = threshold_value
    for i in range(n_quant):
        indexes = np.where(cumsum >= cur_threshold_value)
        z_i_arr[i + 1] = indexes[0][0]
        cur_threshold_value += threshold_value
    z_i_arr[-1] = NORMALIZED
    return z_i_arr


def q_initialization(hist, z_i_arr, n_quant):
    """
    This function compute the initial q array matching the initial segment array
    :param hist: the original histogram to divide into segments
    :param z_i_arr: the initial segment array
    :param n_quant: the amount of segments in the z array
    :return: the initial q array matching the initial segment array
    """
    q_i_arr = np.empty(n_quant).astype(int)
    for i in range(n_quant):
        hist_segment, z_segment = values_from_given_segment(hist, z_i_arr, i)

        # sums up all the p(z) in the range z_i to z_i+1
        denominator = hist_segment.sum()

        # sums up all the z*p(z) in the range z_i to z_i+1
        mult_hist_z = (hist_segment * z_segment).sum()
        q_i_arr[i] = np.round(mult_hist_z / denominator)
    return q_i_arr


def values_from_given_segment(hist, z_i_arr, i):
    """
    This function returns 2 arrays- array of color values from given segment, and
    array of the pixels matching the z values in the z_segment array
    :param hist: the image histogram
    :param z_i_arr: the segment array
    :param i: the i'th segment
    :return: z_segment- array of color values from given segment
            hist_segment- array of the pixels matching the z values in the z_segment
            array
    """
    z_segment = np.arange(z_i_arr[i], z_i_arr[i + 1])
    hist_segment = hist[z_i_arr[i]:z_i_arr[i + 1]]
    return hist_segment, z_segment


def computing_error(z_i_arr, q_i_arr, hist, n_quant):
    """
    This function calculate the error for given z,q arrays
    :param z_i_arr: the segment array
    :param q_i_arr: the q values matching the segment array
    :param hist: the image histogram
    :param n_quant: the number of intensities the output image should have
    :return: the error for given z,q arrays
    """
    err = 0
    for i in range(n_quant):
        hist_segment, z_segment = values_from_given_segment(hist, z_i_arr, i)
        err += ((np.square(q_i_arr[i] - z_segment)) * hist_segment).sum()
    return err


def computing_z_q(n_iter, n_quant, im_orig):
    """
    The main function operating the iterations calculating z,q and errors
    :param n_iter: the maximum number of iterations of the optimization procedure
    :param n_quant: the number of intensities the output image should have
    :param im_orig: the input image
    :return: a list [im_qu, error] where:
            im_qu- the quantized output image
            error - the error array
    """
    hist, bounds = np.histogram(im_orig, bins=np.arange(257))

    # calculate the initial z and q
    z_i_arr = z_initialization(hist, n_quant)
    print(z_i_arr)
    q_i_arr = q_initialization(hist, z_i_arr, n_quant)

    error = []
    for i in range(n_iter):
        z_i_arr_last = z_i_arr.copy()
        z_i_arr = re_calculate_z(n_quant, q_i_arr)
        if np.array_equal(z_i_arr, z_i_arr_last):
            # if z is optimal on the first iteration, calculate one error
            if i == 0:
                error.append(computing_error(z_i_arr, q_i_arr, hist, n_quant))
            break
        q_i_arr = q_initialization(hist, z_i_arr, n_quant)
        error.append(computing_error(z_i_arr, q_i_arr, hist, n_quant))
    im_qu = create_im_using_lut(n_quant, q_i_arr, z_i_arr, im_orig)
    return [im_qu, np.array(error)]


def re_calculate_z(n_quant, q_i_arr):
    """
    this function calculates the z values using the q_array
    :param n_quant: the number of intensities the output image should have
    :param q_i_arr: the q values used to calculate the z values
    :return: the new z values (array)
    """
    z_i_arr_new = np.zeros(n_quant + 1).astype(int)
    z_i_arr_new[0] = 0
    for j in range(1, n_quant):
        z_i_arr_new[j] = np.ceil((q_i_arr[j - 1] + q_i_arr[j]) / 2.0)
    z_i_arr_new[-1] = NORMALIZED
    return z_i_arr_new


def create_im_using_lut(n_quant, q_i_arr, z_i_arr, im_orig):
    """
    This function creates the quantized image using a look up table created from the
    z and q arrays
    :param n_quant: the number of intensities the output image should have
    :param q_i_arr: the q values matching the segment array
    :param z_i_arr: the segment array
    :param im_orig: the input image
    :return: the quantized image
    """
    lut = np.empty(256).astype(int)
    for i in range(n_quant):
        lut[z_i_arr[i]:z_i_arr[i + 1] + 1] = q_i_arr[i]
    im_qu = lut[im_orig.astype(np.uint8)]
    im_qu_float = (im_qu / NORMALIZED).astype(np.float64)
    return im_qu_float


def quantize(im_orig, n_quant, n_iter):
    """
    This function performs optimal quantization of a given grayscale or RGB image
    :param im_orig: the input grayscale or RGB image to be quantized
    :param n_quant: the number of intensities the output image should have
    :param n_iter: the maximum number of iterations of the optimization procedure
    :return: a list [im_qu, error] where:
            im_qu- the quantized output image
            error - the error array
    """
    # if the image is RGB:
    if len(im_orig.shape) == RGB_DIM:
        imYIQ = rgb2yiq(im_orig)
        imYIQ_Y = imYIQ[:, :, 0] * NORMALIZED
        result_list = computing_z_q(n_iter, n_quant, imYIQ_Y)
        imYIQ[:, :, 0] = result_list[0]
        result_list[0] = yiq2rgb(imYIQ)
        return result_list

    # if the image is grayscale:
    else:
        return computing_z_q(n_iter, n_quant, im_orig * NORMALIZED)


if __name__ == '__main__':
    im, hist1, hist2 = histogram_equalize(read_image('jerusalem.jpg', 1))
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()
    quant, err = quantize(im,10,15)

