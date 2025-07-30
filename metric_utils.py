"""
    This code is based on BasicSR's implementations of PSNR and SSIM with
    slight changes.

    Copyright 2018-2022 BasicSR Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


import cv2
import numpy as np
import logging

def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        # Already in range [0, 1]
        if img.max() > 1.001 or img.min() < -0.001 : # Allow for small floating point inaccuracies
             logging.warning(f"Input image is np.float32 but seems to be in range [0, 255]. Dividing by 255.")
             img /= 255.
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace conversion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255]. NOTE: The original docstring description seems slightly
            off, this function expects input range to correspond to the YCbCr calculations
            before scaling back. Let's assume input is float32.
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')

    # Clip the values before converting type or scaling
    if dst_type == np.uint8:
         # Target range is [0, 255]
        img = np.clip(img, 0, 255)
        img = img.round()
    else: # np.float32
        # Target range is [0, 1]. Assuming input calculation resulted in [0, 255] range intermediate.
        img = np.clip(img, 0, 255)
        img /= 255.
        # Optionally clip again after scaling if needed, though previous clip should suffice
        img = np.clip(img, 0, 1)


    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img) # Converts to float32 [0, 1]

    if y_only:
        # Calculate Y channel values in range [16, 235] (for 8-bit)
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        # Calculate YCbCr channels in range [16, 235] for Y, [16, 240] for Cb/Cr
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]

    # Convert back to original type and range
    # The range conversion might need adjustment based on whether the output
    # is expected to be limited-range (16-235/240) or full-range (0-255) for the type.
    # BasicSR's implementation intends to return the same type and range as input.
    # If input was uint8 [0,255], output should be uint8 [0,255] after clipping/rounding.
    # If input was float32 [0,1], output should be float32 [0,1] after scaling/clipping.
    out_img = _convert_output_type_range(out_img, img_type)

    return out_img


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None] # Add channel dimension for grayscale HWC
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0) # CHW to HWC
    # If HWC or grayscale HWC, return as is
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr using the bgr2ycbcr conversion.

    Args:
        img (ndarray): Images with range [0, 255], BGR format (from cv2.imread).

    Returns:
        (ndarray): Y channel image with range [0, 255] (float type) without round.
                   Shape: (h, w, 1)
    """
    # Ensure input is uint8 [0, 255] before conversion if it's not already
    if img.dtype == np.float32 and img.max() <= 1.01:
       img = (img * 255.0).clip(0,255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.clip(0,255).astype(np.uint8) # Handle other types if necessary


    if img.ndim == 3 and img.shape[2] == 3:
        # bgr2ycbcr expects uint8 [0,255] or float32 [0,1]. Let's use uint8.
        img_y = bgr2ycbcr(img, y_only=True) # Output is uint8 [0,255]
        # Convert Y channel back to float32 for calculations within PSNR/SSIM
        img_y = img_y.astype(np.float32) # Range [0, 255] but float type
        if img_y.ndim == 2: # Ensure channel dimension exists
             img_y = img_y[..., None]
    elif img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
         # Already grayscale, assume it's the Y channel
         img_y = img.astype(np.float32) # Ensure float type
         if img_y.ndim == 2:
             img_y = img_y[..., None]
    else:
        raise ValueError(f"Unsupported image shape for Y channel conversion: {img.shape}")

    # The original code returned float * 255, which seems wrong if bgr2ycbcr returns uint8.
    # Let's return float32 in range [0, 255] as expected by PSNR/SSIM calculations.
    return img_y


def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255]. BGR format assumed if color.
        img2 (ndarray): Images with range [0, 255]. BGR format assumed if color.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'. (OpenCV reads as HWC)
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    if not isinstance(img, np.ndarray): img = np.array(img)
    if not isinstance(img2, np.ndarray): img2 = np.array(img2)


    if img.shape != img2.shape:
        raise ValueError(f'Image shapes are different: {img.shape}, {img2.shape}.')


    # Ensure images are in HWC order before processing
    # Note: cv2.imread reads in HWC (or HW for grayscale)
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    # Ensure images are uint8 [0, 255] before potential Y channel conversion
    if img.dtype == np.float32 and img.max() <= 1.01: img = (img * 255.0).clip(0,255).astype(np.uint8)
    elif img.dtype != np.uint8: img = img.clip(0,255).astype(np.uint8)

    if img2.dtype == np.float32 and img2.max() <= 1.01: img2 = (img2 * 255.0).clip(0,255).astype(np.uint8)
    elif img2.dtype != np.uint8: img2 = img2.clip(0,255).astype(np.uint8)


    if test_y_channel:
        # Convert BGR to Y channel (float32 [0, 255], shape HWC with C=1)
        img = to_y_channel(img)
        img2 = to_y_channel(img2)


    # Crop border after potential Y channel conversion
    if crop_border != 0:
        if img.shape[0] <= 2 * crop_border or img.shape[1] <= 2 * crop_border:
             raise ValueError(f"Crop border ({crop_border}) is too large for image dimensions ({img.shape[0]}x{img.shape[1]}).")
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]


    # Calculate PSNR on the potentially Y-channel, cropped image
    # Convert to float64 for precision in MSE calculation
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')

    # PSNR formula assumes max signal value (255 for 8-bit images)
    return 10. * np.log10(255. * 255. / mse)


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Single channel image array with range [0, 255] (float64).
        img2 (ndarray): Single channel image array with range [0, 255] (float64).

    Returns:
        float: SSIM result.
    """
    # Ensure input is float64
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)


    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5) # Gaussian kernel
    window = np.outer(kernel, kernel.transpose()) # 2D Gaussian window

    # Use filter2D for calculating local means, variances, and covariance
    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode (crop boundary effects)
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean() # Average SSIM over the image


def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).

    ``Paper: Image quality assessment: From error visibility to structural similarity``

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged. If test_y_channel is True, it's calculated only on the Y channel.

    Args:
        img (ndarray): Images with range [0, 255]. BGR format assumed if color.
        img2 (ndarray): Images with range [0, 255]. BGR format assumed if color.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: SSIM result.
    """
    if not isinstance(img, np.ndarray): img = np.array(img)
    if not isinstance(img2, np.ndarray): img2 = np.array(img2)

    if img.shape != img2.shape:
         raise ValueError(f'Image shapes are different: {img.shape}, {img2.shape}.')


    # Ensure images are in HWC order
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    # Ensure uint8 for Y channel conversion if needed
    if img.dtype == np.float32 and img.max() <= 1.01: img = (img * 255.0).clip(0,255).astype(np.uint8)
    elif img.dtype != np.uint8: img = img.clip(0,255).astype(np.uint8)

    if img2.dtype == np.float32 and img2.max() <= 1.01: img2 = (img2 * 255.0).clip(0,255).astype(np.uint8)
    elif img2.dtype != np.uint8: img2 = img2.clip(0,255).astype(np.uint8)


    if test_y_channel:
         # Convert to Y channel (float32 [0, 255], shape HWC with C=1)
        img = to_y_channel(img)
        img2 = to_y_channel(img2)
    # Note: If not test_y_channel, SSIM will be calculated per channel (BGR)
    # and averaged. We need float64 for _ssim calculation below.


    # Crop border after potential Y channel conversion
    if crop_border != 0:
        if img.shape[0] <= 2 * crop_border or img.shape[1] <= 2 * crop_border:
             raise ValueError(f"Crop border ({crop_border}) is too large for image dimensions ({img.shape[0]}x{img.shape[1]}).")
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Convert to float64 for SSIM calculation precision AFTER cropping and Y-conversion
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)


    ssims = []
    # Iterate through channels (will be 1 if test_y_channel=True)
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))

    return np.array(ssims).mean() # Return the average SSIM across channels

