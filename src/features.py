
import numpy as np

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
