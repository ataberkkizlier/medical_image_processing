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
import cv2

import numpy as np
import cv2

# 1. Smooth CLAHE 
def smooth_clahe(image, clip_limit=2.0, grid_size=(16, 16)):

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    equalized = clahe.apply(image)

    # Smooth transitions with Gaussian blur
    smooth = cv2.GaussianBlur(equalized, (7, 7), 1.5)

    return smooth


# 2. Smooth Gaussian Blur
def smooth_gaussian_blur(image, kernel_size=7, sigma=1.5):

    k = kernel_size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    # Apply convolution
    padded_image = np.pad(image, k, mode='reflect')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)

    return output


# 3. Refined Adaptive Masking 
def refined_adaptive_masking(image, threshold_factor=0.85):

    # Normalize image to [0, 1]
    normalized = image / 255.0

    # Compute adaptive threshold based on intensity
    mean_intensity = np.mean(normalized)
    threshold = mean_intensity * threshold_factor

    # Generate smooth mask
    mask = cv2.GaussianBlur((normalized > threshold).astype(np.float32), (7, 7), 2)

    # Suppress high-intensity regions
    masked_image = normalized * (1 - mask)

    # Rescale back to 8-bit
    return (masked_image * 255).astype(np.uint8)


# 4. Final Enhancement

def final_enhancement(image, clahe_image, adaptive_masking):

    # Smoothly combine CLAHE and masking
    combined = cv2.addWeighted(clahe_image, 0.8, adaptive_masking, 0.2, 0)

    # Edge Enhancement using Laplacian filter
    laplacian = cv2.Laplacian(combined, cv2.CV_64F)
    enhanced_edges = cv2.addWeighted(combined, 1.0, np.uint8(np.abs(laplacian)), 0.5, 0)

    return enhanced_edges


# 5. Final Refined X-Ray Preprocessing Pipeline
def refined_preprocess_xray(image):

    # 1. Original Image
    original_image = image.copy()

    # 2. Smooth CLAHE
    clahe_image = smooth_clahe(image, clip_limit=2.0, grid_size=(16, 16))

    # 3. Smooth Gaussian Blur
    gaussian_blur = smooth_gaussian_blur(clahe_image, kernel_size=7, sigma=1.5)

    # 4. Refined Adaptive Masking
    adaptive_masking = refined_adaptive_masking(gaussian_blur, threshold_factor=0.85)

    # 5. Final Enhanced Image
    enhanced_image = final_enhancement(gaussian_blur, clahe_image, adaptive_masking)

    # Return results
    return [
        ("Original Image", original_image),
        ("CLAHE", clahe_image),
        ("Gaussian Blur", gaussian_blur),
        ("Adaptive Masking", adaptive_masking),
        ("Final Enhanced", enhanced_image)
    ]


# FILTER 2: Brain MRI Enhancement
import numpy as np
import cv2

# 1. Custom Anisotropic Diffusion
def anisotropic_diffusion(img, num_iter=10, kappa=50, gamma=0.2):
    img = img.astype('float32')
    for _ in range(num_iter):
        nabla_n = np.roll(img, -1, axis=0) - img  # North
        nabla_s = np.roll(img, 1, axis=0) - img   # South
        nabla_e = np.roll(img, -1, axis=1) - img  # East
        nabla_w = np.roll(img, 1, axis=1) - img   # West

        c_n = np.exp(-(nabla_n / kappa) ** 2)
        c_s = np.exp(-(nabla_s / kappa) ** 2)
        c_e = np.exp(-(nabla_e / kappa) ** 2)
        c_w = np.exp(-(nabla_w / kappa) ** 2)

        img += gamma * (
            c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w
        )
    return np.clip(img, 0, 255).astype(np.uint8)


# 2. Smooth CLAHE 
def smooth_clahe(image, clip_limit=2.0, grid_size=(16, 16)):
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    equalized = clahe.apply(image)

    # Smooth transitions with Gaussian blur 
    smooth = cv2.GaussianBlur(equalized, (7, 7), 1.5)

    return smooth


# 3. Smooth Gaussian Blur

def smooth_gaussian_blur(image, kernel_size=7, sigma=1.5):
    k = kernel_size // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    padded_image = np.pad(image, k, mode='reflect')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)

    return output



# 4. Custom Gamma Correction

def custom_gamma_correction(image, gamma=0.5):
    inv_gamma = 1.0 / gamma
    normalized = image / 255.0
    gamma_corrected = np.power(normalized, inv_gamma) * 255
    return gamma_corrected.astype(np.uint8)



# 5. Custom Poisson Noise Simulation

def poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return np.clip(noisy, 0, 255).astype(np.uint8)



# 6. Final Brain MRI Enhancement 

def brain_mri_enhancement(image):
    """
    Custom Brain MRI enhancement pipeline with smooth filters and no grid artifacts.
    """

    # Step 1: Anisotropic Diffusion
    anisotropic = anisotropic_diffusion(image, num_iter=10, kappa=50, gamma=0.2)

    # Step 2: Smooth CLAHE 
    clahe = smooth_clahe(anisotropic, clip_limit=2.0, grid_size=(16, 16))

    # Step 3: Unsharp Masking 
    blurred = smooth_gaussian_blur(clahe, kernel_size=7, sigma=1.5)
    sharpened = np.clip(1.5 * clahe - 0.5 * blurred, 0, 255).astype(np.uint8)

    # Step 4: Poisson Noise Simulation and Removal
    poisson_noisy = poisson_noise(clahe)
    poisson_denoised = anisotropic_diffusion(poisson_noisy, num_iter=5, kappa=50, gamma=0.1)

    # Step 5: Gamma Correction
    gamma_corrected = custom_gamma_correction(poisson_denoised, gamma=0.7)

    # Return all results
    return [
        ("Anisotropic Diffusion", anisotropic),
        ("CLAHE", clahe),
        ("Unsharp Masking", sharpened),
        ("Poisson Noise Removed", poisson_denoised),
        ("Gamma Corrected", gamma_corrected)
    ]


# FILTER 3: Skeleton Enhancement 
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

    # Return all  results
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


# FILTER 4: Vessel Enhancement 
import numpy as np
import cv2


# 1. CLAHE 

def smooth_clahe(image, clip_limit=2.0, grid_size=(8, 8)):

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    result = clahe.apply(image)


    return cv2.GaussianBlur(result, (3, 3), 0)



# 2. Top-hat Preprocessing 

def custom_top_hat(image, kernel_size=(15, 15)):
  
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Use an adaptive kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    # Morphological transformation
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    top_hat = cv2.subtract(image, background)

    # Normalize output
    top_hat = cv2.normalize(top_hat, None, 0, 255, cv2.NORM_MINMAX)

    # Apply light smoothing to avoid noise spikes
    return cv2.GaussianBlur(top_hat, (3, 3), 0)



# 3. Gaussian Smoothing 

def custom_gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Custom Gaussian Blur implementation for artifact smoothing.
    """
    k = kernel_size // 2
    x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    # Apply convolution
    padded_image = np.pad(image, k, mode='reflect')
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(region * kernel)

    return output


# 4. Vessel Mask Extraction 

def vessel_mask_extraction(image, top_hat):

    # Combine CLAHE and Top-hat outputs
    combined = cv2.addWeighted(image, 0.7, top_hat, 0.3, 0)

    # Apply adaptive thresholding
    mask = cv2.adaptiveThreshold(
        combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Apply final mask smoothing
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


# 5. Final Vessel Enhancement Pipeline

def vessel_enhancement(image):

    # Step 1: Convert to grayscale 
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    # Step 2: CLAHE preprocessing
    clahe_image = smooth_clahe(image_gray, clip_limit=2.0, grid_size=(8, 8))

    # Step 3: Top-hat preprocessing
    top_hat_image = custom_top_hat(image_gray, kernel_size=(15, 15))

    # Step 4: Gaussian smoothing
    smooth_image = custom_gaussian_blur(clahe_image, kernel_size=5, sigma=1.0)

    # Step 5: Vessel Mask Extraction
    vessel_mask = vessel_mask_extraction(clahe_image, top_hat_image)

    # Return results
    return [
        ('Original Image', image_gray),
        ('Top-hat Preprocessing', top_hat_image),
        ('CLAHE Preprocessing', clahe_image),
        ('Gaussian Smoothing', smooth_image),
        ('Vessel Mask', vessel_mask)
    ]


# FILTER 5: Brain Tumor Detection 
def brain_tumor_detection(image):

    # 1. Original Image
    original = image.copy()

    # 2. Gaussian Blur
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 1)

    # 3. Total Variation Filtering
    tv_filtered = denoise_tv_chambolle(image / 255.0, weight=0.1)  
    tv_filtered = (tv_filtered * 255).astype(np.uint8)  

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
