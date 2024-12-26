from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
from scipy.signal import convolve2d

app = Flask(__name__, static_folder='static', template_folder='templates')

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

import numpy as np


# ----------------------------
# 1. Custom CLAHE (Localized Contrast Enhancement)
# ----------------------------
def custom_clahe(image, clip_limit=40, grid_size=(8, 8)):
    """
    Custom CLAHE implementation for localized contrast enhancement.
    """
    h, w = image.shape
    tile_h, tile_w = h // grid_size[0], w // grid_size[1]
    result = np.zeros_like(image)

    # Process each tile
    for i in range(0, h, tile_h):
        for j in range(0, w, tile_w):
            # Extract tile
            tile = image[i:i+tile_h, j:j+tile_w]

            # Histogram computation
            hist, bins = np.histogram(tile.flatten(), bins=256, range=[0, 256])

            # Clip histogram
            excess = hist - clip_limit
            excess[excess < 0] = 0
            excess_total = np.sum(excess)
            hist = np.clip(hist, 0, clip_limit)

            # Redistribute excess
            hist += excess_total // 256

            # Compute CDF
            cdf = hist.cumsum()
            cdf_min = cdf.min()
            cdf = (cdf - cdf_min) * 255 / (cdf.max() - cdf_min + 1e-5)  # Avoid div by zero
            cdf = cdf.astype('uint8')

            # Apply equalization
            tile_equalized = cdf[tile]
            result[i:i+tile_h, j:j+tile_w] = tile_equalized

    return result


# ----------------------------
# 2. Custom Gaussian Blur
# ----------------------------
def custom_gaussian_blur(image, kernel_size=3, sigma=1.0):
    """
    Custom Gaussian Blur implementation.
    """
    # Create Gaussian kernel
    k = kernel_size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize kernel

    # Apply convolution
    padded_image = np.pad(image, k, mode='reflect')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)

    return output


# ----------------------------
# 3. Custom Adaptive Masking
# ----------------------------
def custom_adaptive_masking(image):
    """
    Custom Adaptive Masking for highlight suppression.
    """
    # Calculate thresholds
    min_val, max_val = np.min(image), np.max(image)
    threshold = min_val + 0.95 * (max_val - min_val)  # Slightly less aggressive

    # Generate binary mask
    mask = np.where(image > threshold, 255, 0).astype('uint8')

    # Apply mask to suppress highlights
    masked_image = image * (1 - mask // 255)

    return masked_image


# ----------------------------
# 4. Filter 1: Refined Chest X-Ray Preprocessing
# ----------------------------
def refined_preprocess_xray(image):
    """
    Refined preprocessing of chest X-ray images using custom CLAHE, Gaussian Blur, and Adaptive Masking.
    """
    # 1. Original Image
    original_image = image.copy()

    # 2. CLAHE (Custom Implementation)
    clahe_image = custom_clahe(image, clip_limit=40, grid_size=(8, 8))

    # 3. Gaussian Blur (Custom Implementation)
    gaussian_blur = custom_gaussian_blur(clahe_image, kernel_size=3, sigma=1.0)

    # 4. Adaptive Masking (Custom Implementation)
    adaptive_masking = custom_adaptive_masking(image)

    # 5. Final Enhancement: Combine CLAHE + Gaussian + Masking
    combined = gaussian_blur * (1 - adaptive_masking // 255)

    # Return results
    return [
        ("Original Image", original_image),
        ("CLAHE", clahe_image),
        ("Gaussian Blur", gaussian_blur),
        ("Adaptive Masking", adaptive_masking),
        ("Final Enhanced", combined)
    ]

# --- FILTER 2: Brain MRI Enhancement ---
def brain_mri_enhancement(image):
    # 1. Anisotropic Diffusion
    def anisotropic_diffusion(img, num_iter=10, kappa=50, gamma=0.2):
        img = img.astype('float32')
        for _ in range(num_iter):
            nabla = np.gradient(img)
            c = np.exp(-(nabla[0]**2 + nabla[1]**2) / kappa**2)
            img += gamma * np.sum([nabla[0] * c, nabla[1] * c], axis=0)
        return np.clip(img, 0, 255).astype(np.uint8)

    anisotropic = anisotropic_diffusion(image)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(anisotropic)

    # 3. Unsharp Masking (Edge Enhancement)
    blurred = cv2.GaussianBlur(clahe, (5, 5), 0)
    sharpened = cv2.addWeighted(clahe, 1.5, blurred, -0.5, 0)

    # 4. Poisson Noise Simulation and Removal
    poisson_noise = np.random.poisson(clahe / 255.0 * 255).astype('uint8')
    poisson_denoised = denoise_tv_chambolle(poisson_noise / 255.0, weight=0.2)
    poisson_denoised = (poisson_denoised * 255).astype(np.uint8)

    # 5. Gamma Correction
    gamma_corrected = np.array(255 * (poisson_denoised / 255) ** 0.5, dtype='uint8')

    # Return results
    return [
        ("Anisotropic Diffusion", anisotropic),
        ("CLAHE", clahe),
        ("Unsharp Masking", sharpened),
        ("Poisson Noise Removed", poisson_denoised),
        ("Gamma Corrected", gamma_corrected)
    ]

# --- FILTER 3: Skeleton Enhancement ---
def skeleton_enhancement(image):
    """
    Enhance skeleton images using Laplacian, Sobel, smoothing, masking, and gamma correction.
    """
    # 1. Original Image
    original = image.copy()

    # 2. Laplacian Edge Detection
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    # 3. Sharpened Image using Laplacian
    sharpened = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0)

    # 4. Sobel Operator for Edge Enhancement
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.uint8(np.absolute(sobel))

    # 5. Smoothed Sobel
    smooth_sobel = cv2.GaussianBlur(sobel, (5, 5), 0)

    # 6. Masking for Highlighting Features
    mask = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_and(image, image, mask=mask)

    # 7. Gamma Correction for Contrast Adjustment
    gamma_corrected = np.array(255 * (mask / 255) ** 0.5, dtype='uint8')

    # 8. Enhanced Image by Combining Features
    enhanced = cv2.addWeighted(image, 1.0, laplacian, 1.0, 0)

    # Return all intermediate results
    return [
        ("Original Image", original),
        ("Laplacian Image", laplacian),
        ("Sharpened with Laplacian", sharpened),
        ("Sobel Operator", sobel),
        ("Smoothed Sobel", smooth_sobel),
        ("Mask Image", mask),
        ("Enhanced Image", enhanced),
        ("Final Gamma Corrected", gamma_corrected)
    ]


# --- FILTER 4: Vessel Enhancement ---
import cv2
import numpy as np

def vessel_enhancement(image):
    # Check number of channels before converting to grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image  # Already grayscale

    # Proceed with preprocessing steps
    # CLAHE preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image_gray)

    # Top-hat preprocessing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    top_hat_image = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, kernel)

    # Apply smoothing (RGF or other methods)
    smooth_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)

    # Combine all steps and return results
    results = [
    ('Original Image', image_gray),
    ('Top-hat Preprocessing', top_hat_image),
    ('CLAHE Preprocessing', clahe_image),
    ('RGF Smoothing', smooth_image)
]
    return results


from skimage.restoration import denoise_tv_chambolle, unsupervised_wiener, richardson_lucy
from scipy.signal import convolve2d

# --- FILTER 5: Brain Tumor Detection ---
def brain_tumor_detection(image):
    """
    Apply Gaussian Blur, Total Variation Filtering, Wiener Filtering, and Lucy-Richardson Deconvolution
    for Brain Tumor Detection.
    """
    # 1. Original Image
    original = image.copy()

    # 2. Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 1)

    # 3. Total Variation Filtering
    tv_filtered = denoise_tv_chambolle(image / 255.0, weight=0.1)  # Normalize image
    tv_filtered = (tv_filtered * 255).astype(np.uint8)  # Convert back to uint8

    # 4. Wiener Filtering
    psf = np.ones((5, 5)) / 25  # Point Spread Function
    image_conv = convolve2d(image, psf, 'same')
    wiener_filtered, _ = unsupervised_wiener(image_conv, psf)
    wiener_filtered = np.clip(wiener_filtered, 0, 255).astype(np.uint8)  # Clip and convert to uint8

    # 5. Lucy-Richardson Deconvolution
    psf = np.ones((5, 5)) / 25  # Point Spread Function for Richardson-Lucy
    lucy_richardson = richardson_lucy(image / 255.0, psf, num_iter=10)  # Deconvolution
    lucy_richardson = (lucy_richardson * 255).astype(np.uint8)  # Convert back to uint8

    # Return results
    return [
        ("Original Image", original),
        ("Gaussian Blur", gaussian_blur),
        ("Total Variation Filtering", tv_filtered),
        ("Wiener Filtering", wiener_filtered),
        ("Lucy-Richardson Deconvolution", lucy_richardson)
    ]


# --- Home Page ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# --- Process Image ---
@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image']
    filter_type = request.form['filter_type']
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({"error": "Invalid image file."}), 400

    # Apply selected filter
    if filter_type == 'refined_xray':
        results = refined_preprocess_xray(image)
    elif filter_type == 'brain_mri':
        results = brain_mri_enhancement(image)
    elif filter_type == 'skeleton':
        results = skeleton_enhancement(image)
    elif filter_type == 'vessels':
        results = vessel_enhancement(image)
    elif filter_type == 'brain_tumor_detection':
        results = brain_tumor_detection(image)

    else:
        return jsonify({"error": "Invalid filter type."}), 400

    # Save and return results
    result_paths = []
    for idx, (title, img) in enumerate(results):
        result_path = os.path.join(RESULTS_FOLDER, f"{filename}_{title.replace(' ', '_')}.png")
        cv2.imwrite(result_path, img)
        result_paths.append((title, f'/results/{filename}_{title.replace(" ", "_")}.png'))

    return jsonify({"image_urls": result_paths})


# --- Serve Results ---
@app.route('/results/<path:filename>')
def get_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
