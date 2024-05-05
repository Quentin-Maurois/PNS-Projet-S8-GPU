import argparse
from numba import cuda
from PIL import Image
import numpy as np
import math

# Code by Quentin Maurois
# This code is a GPU implementation of the kernels used in the Canny edge detection algorithm.
#
# Some naming conventions:
# The arrays suffixed with "_cuda" to indicate that they are device arrays.
# The start of the kernel names indicate the operation that they perform or will soon perform.
# ex: "bw_image_array_cuda" is the array that will store the black and white image on the device.
#
# The kernels used in this project are:
# bw_kernel: Converts an image to black and white.
# gauss_kernel: Applies a Gaussian blur to an image.
# sobel_kernel: Applies Sobel edge detection to an image.
# threshold_kernel: Applies a threshold to an image.
# hysteresis_kernel: Applies hysteresis to an image.
#
# The kernels can only be called in this order and using an original image as an input.


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process input image using GPU operations.')

    # Add optional arguments
    parser.add_argument('--tb', type=int, help='Optional size of a thread block for all operations')
    parser.add_argument('--bw', action='store_true', help='perform only the bw_kernel')
    parser.add_argument('--gauss', action='store_true', help='perform the bw_kernel and the gauss_kernel')
    parser.add_argument('--sobel', action='store_true', help='perform all kernels up to sobel_kernel  and write to disk the magnitude of each pixel')
    parser.add_argument('--threshold', action='store_true', help='perform all kernels up to threshold_kernel')

    # Add positional arguments
    parser.add_argument('inputImage', type=str, help='Input image file path.')
    parser.add_argument('outputImage', type=str, help='Output image file path.')

    # Parse the arguments
    args = parser.parse_args()

    # Retrieve arguments
    tb_value = args.tb
    bw_enabled = args.bw
    gauss_enabled = args.gauss
    sobel_enabled = args.sobel
    threshold_enabled = args.threshold
    input_image = args.inputImage
    output_image = args.outputImage

    # Opening the image
    input_img = Image.open(input_image)
    input_img_array = np.array(input_img)

    # Set the thread block size
    block_size = (16, 16)
    # Correct the block size if tb_value is provided
    if tb_value:
        block_size = (tb_value, tb_value)
    # Set the grid size
    grid_size = ((input_img_array.shape[1] - 1) // block_size[0] + 1,  # Corrected dimension
                 (input_img_array.shape[0] - 1) // block_size[1] + 1)  # Corrected dimension


    # Convert the image to black and white
    input_image_array_cuda = cuda.to_device(np.ascontiguousarray(input_img_array))
    bw_image_array_cuda = cuda.device_array_like(input_img_array)
    bw_kernel[grid_size, block_size](input_image_array_cuda, bw_image_array_cuda)

    if bw_enabled:
        bw_image_array = bw_image_array_cuda.copy_to_host()
        bw_image = Image.fromarray(bw_image_array)
        bw_image.save(output_image)
        return


    # Apply Gaussian blur
    gaussian_kernel = np.array([[1, 4, 6, 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]], dtype=np.float32)  # Ensure dtype is float32
    gauss_image_array_cuda = cuda.device_array_like(input_img_array)
    gauss_kernel[grid_size, block_size](bw_image_array_cuda, gauss_image_array_cuda, gaussian_kernel)

    if gauss_enabled:
        gauss_image_array = gauss_image_array_cuda.copy_to_host()
        gauss_image = Image.fromarray(gauss_image_array)
        gauss_image.save(output_image)
        return


    # Apply Sobel edge detection
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)  # Ensure dtype is float32
    sobel_kernel_y = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=np.float32)  # Ensure dtype is float32
    sobel_magnitude_array_cuda = cuda.device_array_like(input_img_array)
    sobel_theta_array_cuda = cuda.device_array_like(input_img_array)
    sobel_x_array_cuda = cuda.device_array_like(input_img_array)
    sobel_y_array_cuda = cuda.device_array_like(input_img_array)
    sobel_kernel[grid_size, block_size](gauss_image_array_cuda, sobel_x_array_cuda, sobel_kernel_x, sobel_y_array_cuda, sobel_kernel_y, sobel_magnitude_array_cuda, sobel_theta_array_cuda)

    if sobel_enabled:
        sobel_magnitude_array = sobel_magnitude_array_cuda.copy_to_host()
        sobel_magnitude_image = Image.fromarray(sobel_magnitude_array)
        sobel_magnitude_image.save(output_image)
        return
    

    # Apply thresholding
    threshold_value = 100
    threshold_image_array_cuda = cuda.device_array_like(input_img_array)
    threshold_kernel[grid_size, block_size](sobel_magnitude_array_cuda, threshold_image_array_cuda, threshold_value)

    if threshold_enabled:
        threshold_image_array = threshold_image_array_cuda.copy_to_host()
        threshold_image = Image.fromarray(threshold_image_array)
        threshold_image.save(output_image)
        return
    

    # Apply hysteresis
    low_threshold = 50
    high_threshold = 255
    hysteresis_image_array_cuda = cuda.device_array_like(input_img_array)
    hysteresis_kernel[grid_size, block_size](threshold_image_array_cuda, hysteresis_image_array_cuda, low_threshold, high_threshold)

    hysteresis_image_array = hysteresis_image_array_cuda.copy_to_host()
    hysteresis_image = Image.fromarray(hysteresis_image_array)
    hysteresis_image.save(output_image)







@cuda.jit
def bw_kernel(input_array, output_array):
    x, y = cuda.grid(2)
    if y < input_array.shape[0]:
        if x < input_array.shape[1]:
            r, g, b = input_array[y, x]
            bw_value = 0.2989 * r + 0.5870 * g + 0.1140 * b
            output_array[y, x] = int(math.ceil(bw_value))


@cuda.jit
def gauss_kernel(input_array, output_array, kernel):
    x, y = cuda.grid(2)
    if y < input_array.shape[0]:  # Corrected dimension
        if x < input_array.shape[1]:  # Corrected dimension
            for c in range(input_array.shape[2]):
                kernel_sum = 0.0  # Use floating-point types
                weighted_sum = 0.0  # Use floating-point types
                for a in range(-2, 3):
                    for b in range(-2, 3):
                        ny = min(max(0, y + a), input_array.shape[0] - 1)
                        nx = min(max(0, x + b), input_array.shape[1] - 1)
                        weight = kernel[a + 2, b + 2]  # Correct indexing
                        kernel_sum += weight
                        weighted_sum += input_array[ny, nx, c] * weight
                if kernel_sum != 0.0:  # Avoid division by zero
                    output_array[y, x, c] = int(math.ceil(weighted_sum / kernel_sum))  # Use floating-point division
                else:
                    output_array[y, x, c] = int(math.ceil(input_array[y, x, c]))  # Preserve original pixel value if division by zero


@cuda.jit
def sobel_kernel(input_array, output_array_x, sobel_kernel_x, output_array_y, sobel_kernel_y, magnitude_array, theta_array):
    x, y = cuda.grid(2)
    if y < input_array.shape[0]:
        if x < input_array.shape[1]:
            for c in range(input_array.shape[2]):
                gradient_x = 0.0
                gradient_y = 0.0
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        ny = min(max(0, y + a), input_array.shape[0] - 1)
                        nx = min(max(0, x + b), input_array.shape[1] - 1)
                        weight_x = sobel_kernel_x[a + 1, b + 1]
                        weight_y = sobel_kernel_y[a + 1, b + 1]
                        gradient_x += input_array[ny, nx, c] * weight_x
                        gradient_y += input_array[ny, nx, c] * weight_y
                output_array_x[y, x, c] = int(math.ceil(gradient_x))
                output_array_y[y, x, c] = int(math.ceil(gradient_y))
                magnitude_array[y, x, c] = int(math.ceil(abs(gradient_x) + abs(gradient_y)))
                theta_array[y, x, c] = int(math.ceil(math.atan2(gradient_y, gradient_x)))


@cuda.jit
def threshold_kernel(input_array, output_array, threshold):
    x, y = cuda.grid(2)
    if y < input_array.shape[0]:
        if x < input_array.shape[1]:
            for c in range(input_array.shape[2]):
                if input_array[y, x, c] > threshold:
                    output_array[y, x, c] = 255
                else:
                    output_array[y, x, c] = 0
    

@cuda.jit
def hysteresis_kernel(input_array, output_array, low_threshold, high_threshold):
    x, y = cuda.grid(2)
    if y < input_array.shape[0]:
        if x < input_array.shape[1]:
            for c in range(input_array.shape[2]):
                if input_array[y, x, c] >= high_threshold:
                    output_array[y, x, c] = 255
                elif input_array[y, x, c] < low_threshold:
                    output_array[y, x, c] = 0
                else:
                    # Check 8-connected neighbors for strong edges
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            ny = y + i
                            nx = x + j
                            if ny >= 0 and ny < input_array.shape[0] and nx >= 0 and nx < input_array.shape[1]:
                                if input_array[ny, nx, c] >= high_threshold:
                                    output_array[y, x, c] = 255
                                    break
                    else:
                        output_array[y, x, c] = 0
                       

if __name__ == "__main__":
    main()