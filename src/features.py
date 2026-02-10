
import numpy as np
import cv2

def first_order_features(gray_img):
    """
    Extracts first-order texture features from a grayscale image.
    Returns a list: [mean, variance, skewness, kurtosis, energy, entropy]
    """
    flat = gray_img.flatten()
    # Normalize to 0-255 integers if not already? assuming input is uint8 or similar range
    # The notebook code assumes flat is discrete values for bincount
    
    # Check if image needs casting
    if flat.dtype.kind == 'f':
        # If float, maybe map to 0-255? The notebook does this in main:
        # gray = (gray * 255).astype(np.uint8)
        # We assume input is already suitable for bincount (int)
        pass 
        
    hist = np.bincount(flat, minlength=256) / len(flat)
    indices = np.arange(256)
    mean = np.sum(indices * hist)
    variance = np.sum((indices - mean)**2 * hist)
    
    if variance == 0:
        skewness = 0
        kurtosis = 0
    else:
        skewness = np.sum(((indices - mean) / np.sqrt(variance))**3 * hist)
        kurtosis = np.sum(((indices - mean) / np.sqrt(variance))**4 * hist) - 3
        
    energy = np.sum(hist**2)
    # Avoid log(0)
    entropy = -np.sum(hist * np.log2(hist + (hist == 0)))
    
    return [mean, variance, skewness, kurtosis, energy, entropy]

def calculate_glcm(gray_img, dx, dy, levels):
    """
    Computes the GLCM for a given direction (dx, dy).
    """
    glcm = np.zeros((levels, levels), dtype=np.float64)
    height, width = gray_img.shape
    
    # The notebook implementation of GLCM
    start_i = max(0, -dy)
    end_i = height - max(0, dy)
    start_j = max(0, -dx)
    end_j = width - max(0, dx)
    
    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            i2 = i + dy
            j2 = j + dx
            val1 = gray_img[i, j]
            val2 = gray_img[i2, j2]
            glcm[val1, val2] += 1
            
    glcm += glcm.T
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm

def glcm_features(glcm_avg, levels):
    """
    Extracts 14 GLCM features from the averaged GLCM matrix.
    """
    p = glcm_avg
    i, j = np.ogrid[0:levels, 0:levels]
    p_x = np.sum(p, axis=1)
    p_y = np.sum(p, axis=0)
    
    mu_x = np.sum(i[:, 0] * p_x)
    mu_y = np.sum(j[0, :] * p_y)
    
    sigma_x_sq = np.sum((i[:, 0] - mu_x)**2 * p_x)
    sigma_y_sq = np.sum((j[0, :] - mu_y)**2 * p_y)
    
    sigma_x = np.sqrt(sigma_x_sq)
    sigma_y = np.sqrt(sigma_y_sq)

    # 1. ASM
    asm = np.sum(p ** 2)

    # 2. Contrast
    contrast = np.sum(p * (i - j)**2)

    # 3. Correlation
    if sigma_x * sigma_y == 0:
        correlation = 1
    else:
        correlation = np.sum(p * (i - mu_x) * (j - mu_y)) / (sigma_x * sigma_y)

    # 4. Variance
    variance = np.sum(p * (i - mu_x)**2)

    # 5. IDM / Homogeneity
    idm = np.sum(p / (1 + (i - j)**2))

    # Helpers for sum/diff features
    max_sum = 2 * (levels - 1)
    p_xplusy = np.zeros(max_sum + 1)
    for ii in range(levels):
        for jj in range(levels):
            p_xplusy[ii + jj] += p[ii, jj]

    # 6. Sum Average
    sum_avg = np.sum(np.arange(max_sum + 1) * p_xplusy)

    # 7. Sum Variance
    sum_var = np.sum((np.arange(max_sum + 1) - sum_avg)**2 * p_xplusy)

    # 8. Sum Entropy
    sum_ent = -np.sum(p_xplusy * np.log2(p_xplusy + (p_xplusy == 0)))

    # 9. Entropy
    ent = -np.sum(p * np.log2(p + (p == 0)))

    # Helper for diff features
    max_diff = levels - 1
    p_xminusy = np.zeros(max_diff + 1)
    for ii in range(levels):
        for jj in range(levels):
            p_xminusy[abs(ii - jj)] += p[ii, jj]

    # 10. Difference Variance
    diff_mean = np.sum(np.arange(max_diff + 1) * p_xminusy)
    diff_var = np.sum(np.arange(max_diff + 1)**2 * p_xminusy) - diff_mean**2

    # 11. Difference Entropy
    diff_ent = -np.sum(p_xminusy * np.log2(p_xminusy + (p_xminusy == 0)))

    # 12. IMC1, 13. IMC2
    HX = -np.sum(p_x * np.log2(p_x + (p_x == 0)))
    HY = -np.sum(p_y * np.log2(p_y + (p_y == 0)))
    HXY = ent
    
    # Small epsilon to avoid log(0) in cross terms if needed, though + 1e-10 is used in notebook
    HXY1 = -np.sum(p * np.log2(p_x[:, np.newaxis] * p_y[np.newaxis, :] + 1e-10))
    HXY2 = -np.sum(p_x[:, np.newaxis] * p_y[np.newaxis, :] * np.log2(p_x[:, np.newaxis] * p_y[np.newaxis, :] + 1e-10))
    
    if max(HX, HY) == 0:
        imc1 = 0
    else:
        imc1 = (HXY - HXY1) / max(HX, HY)
        
    # Validating sqrt content for IMC2
    imc2_term = 1 - np.exp(-2 * (HXY2 - HXY))
    imc2 = np.sqrt(max(0, imc2_term))

    # 14. MCC
    # Construct Q matrix
    Q = np.zeros((levels, levels))
    # Handling division by zero if p_x is 0
    with np.errstate(divide='ignore', invalid='ignore'):
         for ii in range(levels):
            if p_x[ii] == 0:
                continue
            for jj in range(levels):
                Q[ii, jj] = np.dot(p[ii, :], p[jj, :] / (p_x + 1e-10)) / p_x[ii]
                
    eigs = np.linalg.eigvals(Q)
    # Sort eigenvalues
    sorted_eigs = np.sort(np.real(eigs))[::-1]
    
    mcc = np.sqrt(sorted_eigs[1]) if len(sorted_eigs) > 1 and sorted_eigs[1] >= 0 else 0

    return [asm, contrast, correlation, variance, idm, sum_avg, sum_var, sum_ent, ent, diff_var, diff_ent, imc1, imc2, mcc]


def _quantize_to_levels(gray_u8, levels):
    """Map uint8 [0..255] -> uint8 [0..levels-1]."""
    if levels <= 1:
        return np.zeros_like(gray_u8, dtype=np.uint8)
    # Use floor mapping for stable bins
    q = (gray_u8.astype(np.int32) * levels) // 256
    q = np.clip(q, 0, levels - 1).astype(np.uint8)
    return q

def _average_glcm_over_offsets(patch_q, offsets, levels):
    """Average symmetric GLCM over a list of (dx,dy) offsets."""
    glcms = []
    for dx, dy in offsets:
        glcms.append(calculate_glcm(patch_q, dx=dx, dy=dy, levels=levels))
    glcm_avg = np.mean(glcms, axis=0) if len(glcms) else np.zeros((levels, levels), dtype=np.float64)
    # Re-normalize
    s = glcm_avg.sum()
    if s > 0:
        glcm_avg = glcm_avg / s
    return glcm_avg

def texture_feature_maps(gray_img_u8, win=32, step=8, levels=32, offsets=None, features_to_map=None):
    """
    Compute local texture feature maps by sliding-window GLCM analysis.
    
    Returns:
        fmap_dict: dict[str, 2D float array]
        meta: dict with coords and coverage
    """
    if offsets is None:
        offsets = [(1, 0), (1, -1), (0, -1), (-1, -1)]
    if features_to_map is None:
        features_to_map = ['Contrast', 'Homogeneity', 'Entropy_GLCM', 'Entropy']

    H, W = gray_img_u8.shape
    ys = list(range(0, H - win + 1, step))
    xs = list(range(0, W - win + 1, step))
    if len(ys) == 0 or len(xs) == 0:
        return {}, {"ys": ys, "xs": xs, "covered_w": 0, "covered_h": 0}

    covered_w = xs[-1] + win
    covered_h = ys[-1] + win

    # Prepare outputs
    fmap_dict = {k: np.zeros((len(ys), len(xs)), dtype=np.float32) for k in features_to_map}

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            patch = gray_img_u8[y:y+win, x:x+win]

            # First-order features
            fo = first_order_features(patch)
            fo_names = ['Mean', 'Variance', 'Skewness', 'Kurtosis', 'Energy', 'Entropy']
            fo_map = dict(zip(fo_names, fo))

            # GLCM features
            patch_q = _quantize_to_levels(patch, levels)
            glcm_avg = _average_glcm_over_offsets(patch_q, offsets, levels)
            gl = glcm_features(glcm_avg, levels)
            gl_names = ['ASM', 'Contrast', 'Correlation', 'Variance_GLCM', 'Homogeneity', 
                        'Sum_Average', 'Sum_Variance', 'Sum_Entropy', 'Entropy_GLCM', 
                        'Difference_Variance', 'Difference_Entropy', 'IMC1', 'IMC2', 'MCC']
            gl_map = dict(zip(gl_names, gl))

            for name in features_to_map:
                if name in fo_map:
                    fmap_dict[name][iy, ix] = fo_map[name]
                elif name in gl_map:
                    fmap_dict[name][iy, ix] = gl_map[name]

    meta = {"ys": ys, "xs": xs, "covered_w": covered_w, "covered_h": covered_h}
    return fmap_dict, meta

def extract_all_features(gray_img, levels=32):
    """
    Wrapper to extract both first-order and GLCM features.
    
    1. First-Order (6 features)
    2. GLCM (14 features) -> Averaged over 4 directions
    
    Returns:
        dict: feature_name -> value
    """
    
    # 1. First Order
    fo_vals = first_order_features(gray_img)
    fo_names = ['Mean', 'Variance', 'Skewness', 'Kurtosis', 'Energy', 'Entropy']
    
    # 2. GLCM
    # Quantize image for GLCM
    # assuming gray_img is 0-255
    gray_quant = (gray_img // (256 // levels)).astype(int)
    
    directions = [(1, 0), (1, 1), (0, 1), (-1, 1)]
    glcms = [calculate_glcm(gray_quant, dx, dy, levels) for dx, dy in directions]
    glcm_avg = np.mean(glcms, axis=0)
    
    glcm_vals = glcm_features(glcm_avg, levels)
    glcm_names = ['ASM', 'Contrast', 'Correlation', 'Variance_GLCM', 'Homogeneity', 
                  'Sum_Average', 'Sum_Variance', 'Sum_Entropy', 'Entropy_GLCM', 
                  'Difference_Variance', 'Difference_Entropy', 'IMC1', 'IMC2', 'MCC']
    
    features = {}
    for n, v in zip(fo_names, fo_vals):
        features[n] = v
    for n, v in zip(glcm_names, glcm_vals):
        features[n] = v
        
    return features

# --- Heatmap Generation ---
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.ndimage import generic_filter


def generate_texture_heatmap(gray_img, method='entropy', disk_size=3, win=32, step=8, levels=32):
    """
    Generates a normalized texture heatmap from a grayscale image.
    
    Args:
        gray_img (numpy.ndarray): Input grayscale image.
        method (str): 'entropy', 'std', or 'sliding_window'.
        disk_size (int): Radius for 'entropy' and 'std' methods.
        win (int): Window size for 'sliding_window'.
        step (int): Step size for 'sliding_window'.
        levels (int): GLCM levels for 'sliding_window'.
    """
    
    if method == 'sliding_window':
        # Default to Entropy_GLCM as the primary "complexity" measure for sliding window
        fmap_dict, meta = texture_feature_maps(gray_img, win=win, step=step, levels=levels, features_to_map=['Entropy_GLCM'])
        if 'Entropy_GLCM' in fmap_dict:
            raw_map = fmap_dict['Entropy_GLCM']
            # Resize back to original image size for overlay
            texture_map = cv2.resize(raw_map, (gray_img.shape[1], gray_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            texture_map = np.zeros_like(gray_img, dtype=np.float32)

    elif method == 'entropy':
        # Entropy filter requires uint8
        if gray_img.dtype != np.uint8:
            img_uint8 = (gray_img).astype(np.uint8)
        else:
            img_uint8 = gray_img
            
        # Compute local entropy
        texture_map = entropy(img_uint8, disk(disk_size))
        
    elif method == 'std':
        # Local Standard Deviation
        # Using a faster approximation or generic_filter
        texture_map = generic_filter(gray_img, np.std, size=2*disk_size+1)
        
    else:
        # Default fallback
        texture_map = np.zeros_like(gray_img, dtype=np.float32)

    # Normalize to 0-1 for visualization
    if texture_map.max() > texture_map.min():
        texture_map = (texture_map - texture_map.min()) / (texture_map.max() - texture_map.min())
    else:
        texture_map = np.zeros_like(texture_map)
        
    return texture_map
