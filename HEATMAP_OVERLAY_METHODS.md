# Texture Analysis Heatmap Overlay Methods
## Comprehensive Guide for Medical Image Analysis

**Date:** February 9, 2026  
**Project:** Mayo Endoscopic Score Texture Analysis  
**Purpose:** Methods for creating informative heatmap overlays from colonoscopy images

---

## Table of Contents
1. [Current Implementation Analysis](#current-implementation-analysis)
2. [Additional Per-Pixel Methods](#additional-per-pixel-methods)
3. [Additional Per-Block Methods](#additional-per-block-methods)
4. [Frequency & Multi-Scale Methods](#frequency--multi-scale-methods)
5. [Color-Based Methods](#color-based-methods)
6. [Trade-offs & Performance](#trade-offs--performance)
7. [Medical Imaging Recommendations](#medical-imaging-recommendations)
8. [Implementation Examples](#implementation-examples)
9. [Best Practices](#best-practices)

---

## Current Implementation Analysis

### Overview
Your notebooks already implement a sophisticated **sliding window texture analysis** framework that generates high-quality heatmaps using block-based (patch-based) feature extraction.

### Configuration Parameters
```python
WIN = 32          # Window/patch size (pixels)
STEP = 8          # Stride/overlap (pixels)
GLCM_LEVELS = 32  # Quantization levels for GLCM
OFFSETS = [(1,0), (1,-1), (0,-1), (-1,-1)]  # 0°, 45°, 90°, 135°
```

### Currently Implemented Features

#### 1. First-Order Statistical Features (Fast, O(n²))
These analyze the intensity distribution within each patch:

| Feature | Description | Clinical Relevance |
|---------|-------------|-------------------|
| **Mean** | Average pixel intensity | Overall brightness, general inflammation level |
| **Variance** | Intensity spread/variability | Surface roughness, texture heterogeneity |
| **Skewness** | Asymmetry of distribution | Directional texture bias |
| **Kurtosis** | "Peakedness" of distribution | Outlier presence, extreme values |
| **Energy** | Sum of squared histogram values | Texture uniformity/orderliness |
| **Entropy** | Randomness/disorder measure | Texture complexity, disorganization |

**Computational Cost:** Very low (~0.001s per 32×32 patch)  
**Memory Usage:** Minimal  
**Advantages:** 
- Extremely fast computation
- Intuitive interpretation
- Low noise sensitivity
- Good for global texture changes

**Limitations:**
- No spatial relationship information
- Cannot detect directional patterns
- Limited discriminative power for similar textures

#### 2. GLCM (Gray-Level Co-occurrence Matrix) Features (Moderate, O(n²×L²))

These capture spatial relationships between pixel pairs:

| Feature | Description | What It Detects | Mayo Score Utility |
|---------|-------------|-----------------|-------------------|
| **Contrast** | Local intensity variations | Texture roughness, mucosal irregularity | ⭐⭐⭐⭐⭐ High |
| **Homogeneity** | Texture smoothness/uniformity | Smooth vs. irregular mucosa | ⭐⭐⭐⭐⭐ High |
| **Correlation** | Linear gray-tone dependencies | Directional patterns | ⭐⭐⭐ Medium |
| **ASM** | Angular Second Moment | Texture orderliness | ⭐⭐⭐ Medium |
| **GLCM Entropy** | Co-occurrence randomness | Texture complexity | ⭐⭐⭐⭐ High |
| **Sum Average** | Weighted average of co-occurrence | Overall texture brightness | ⭐⭐ Low |
| **Sum Variance** | Variance of sum distribution | Texture spread | ⭐⭐⭐ Medium |
| **Sum Entropy** | Entropy of sum distribution | Combined texture randomness | ⭐⭐⭐ Medium |
| **Difference Variance** | Variance of difference distribution | Local variation strength | ⭐⭐⭐⭐ High |
| **Difference Entropy** | Entropy of difference distribution | Edge/boundary complexity | ⭐⭐⭐⭐ High |
| **IMC1, IMC2** | Information Measure of Correlation | Texture dependency strength | ⭐⭐ Low |
| **MCC** | Maximal Correlation Coefficient | Complex texture correlation | ⭐⭐ Low |

**Computational Cost:** Moderate (~0.01-0.05s per 32×32 patch)  
**Memory Usage:** Moderate (depends on quantization levels)  

**Advantages:**
- Captures spatial texture relationships
- Rotation-invariant (when averaged over directions)
- Proven effectiveness in medical imaging
- Rich discriminative information

**Limitations:**
- Slower than first-order features
- Requires quantization (information loss)
- Sensitive to parameter selection (distances, angles, levels)

---

## Additional Per-Pixel Methods

These generate smooth, fine-grained heatmaps by computing features at every pixel location.

### 1. Local Standard Deviation

**Concept:** Compute standard deviation within a small neighborhood around each pixel.

```python
from scipy.ndimage import generic_filter

def local_std_map(gray_img, kernel_size=7):
    """
    Compute per-pixel standard deviation within local neighborhoods.
    
    Parameters:
    -----------
    gray_img : ndarray (H, W)
        Grayscale image (0-255 or 0.0-1.0)
    kernel_size : int
        Size of local neighborhood (odd number recommended)
        
    Returns:
    --------
    std_map : ndarray (H, W)
        Standard deviation at each pixel
    """
    return generic_filter(gray_img, np.std, size=kernel_size)

# Or faster implementation with convolution:
def local_std_map_fast(gray_img, kernel_size=7):
    from scipy.ndimage import uniform_filter
    
    mean = uniform_filter(gray_img.astype(float), kernel_size)
    mean_sq = uniform_filter(gray_img.astype(float)**2, kernel_size)
    variance = mean_sq - mean**2
    return np.sqrt(np.maximum(variance, 0))  # Ensure non-negative
```

**Use Cases:**
- Surface roughness quantification
- Inflammation detection (rough vs. smooth mucosa)
- Vascular pattern prominence
- Ulceration boundary detection

**Computational Cost:** Fast (~0.1s for 512×512 image)  
**Clinical Value:** ⭐⭐⭐⭐⭐

---

### 2. Gradient Magnitude (Edge Strength)

**Concept:** Measure rate of intensity change at each pixel.

```python
from scipy.ndimage import sobel

def gradient_magnitude_map(gray_img):
    """
    Compute edge strength at each pixel using Sobel operator.
    
    Returns:
    --------
    gradient_mag : ndarray (H, W)
        Magnitude of intensity gradient
    """
    # Sobel filters in x and y directions
    grad_x = sobel(gray_img, axis=1)
    grad_y = sobel(gray_img, axis=0)
    
    # Magnitude of gradient vector
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    return gradient_mag

def gradient_orientation_map(gray_img):
    """
    Compute gradient direction at each pixel.
    
    Returns:
    --------
    orientation : ndarray (H, W)
        Angle of gradient in radians [-π, π]
    """
    grad_x = sobel(gray_img, axis=1)
    grad_y = sobel(gray_img, axis=0)
    
    return np.arctan2(grad_y, grad_x)
```

**Extended Version - Edge Density:**
```python
def edge_density_map(gray_img, threshold_percentile=75, block_size=16):
    """
    Compute density of edge pixels in local neighborhoods.
    
    Useful for detecting vascular patterns and mucosal complexity.
    """
    gradient_mag = gradient_magnitude_map(gray_img)
    
    # Threshold to binary edge map
    threshold = np.percentile(gradient_mag, threshold_percentile)
    edges = (gradient_mag > threshold).astype(float)
    
    # Compute local density using uniform filter
    from scipy.ndimage import uniform_filter
    edge_density = uniform_filter(edges, size=block_size)
    
    return edge_density
```

**Use Cases:**
- Vascular pattern detection
- Mucosal fold prominence
- Ulcer edge definition
- Texture boundary identification

**Computational Cost:** Very fast (~0.05s for 512×512 image)  
**Clinical Value:** ⭐⭐⭐⭐

---

### 3. Local Binary Patterns (LBP)

**Concept:** Encode local texture patterns as binary numbers by comparing each pixel with its neighbors.

```python
from skimage.feature import local_binary_pattern

def lbp_map(gray_img, P=8, R=1, method='uniform'):
    """
    Compute Local Binary Pattern for each pixel.
    
    Parameters:
    -----------
    P : int
        Number of circularly symmetric neighbor points
    R : float
        Radius of circle (pixel distance)
    method : str
        'uniform' - Rotation-invariant uniform patterns (59 bins for P=8)
        'default' - All possible patterns (2^P bins)
        'ror' - Rotation-invariant (non-uniform)
        'var' - Rotation-invariant variance
        
    Returns:
    --------
    lbp : ndarray (H, W)
        LBP code at each pixel
    """
    return local_binary_pattern(gray_img, P, R, method=method)

def lbp_variance_map(gray_img, P=8, R=1):
    """
    Compute LBP variance - rotation-invariant texture measure.
    Captures contrast/roughness at each pixel.
    """
    return local_binary_pattern(gray_img, P, R, method='var')
```

**Advanced Usage - Multi-Scale LBP:**
```python
def multiscale_lbp_map(gray_img, radii=[1, 2, 3]):
    """
    Combine LBP at multiple scales for richer texture description.
    """
    lbp_maps = []
    for R in radii:
        P = 8 * R  # Adjust number of points with radius
        lbp = local_binary_pattern(gray_img, P, R, method='uniform')
        lbp_maps.append(lbp)
    
    # Combine (e.g., average or concatenate histograms)
    return np.mean(lbp_maps, axis=0)
```

**Use Cases:**
- Micro-texture pattern recognition
- Rotation-invariant texture classification
- Fine-grained surface characterization
- Discriminating similar-looking textures

**Computational Cost:** Fast (~0.2s for 512×512 image)  
**Clinical Value:** ⭐⭐⭐⭐

---

### 4. Local Range (Max - Min)

**Concept:** Difference between maximum and minimum intensities in local window.

```python
from scipy.ndimage import maximum_filter, minimum_filter

def local_range_map(gray_img, size=7):
    """
    Compute local intensity range (max - min) at each pixel.
    
    Simple alternative to standard deviation, more robust to outliers.
    """
    local_max = maximum_filter(gray_img, size=size)
    local_min = minimum_filter(gray_img, size=size)
    return local_max - local_min
```

**Use Cases:**
- Similar to local std but more robust
- Captures local contrast
- Less sensitive to noise

**Computational Cost:** Very fast (~0.05s for 512×512 image)  
**Clinical Value:** ⭐⭐⭐⭐

---

### 5. Entropy Filter

**Concept:** Shannon entropy computed in local neighborhoods.

```python
from skimage.filters.rank import entropy
from skimage.morphology import disk

def local_entropy_map(gray_img, radius=3):
    """
    Compute local Shannon entropy at each pixel.
    
    Requires uint8 input for skimage.filters.rank
    """
    if gray_img.dtype != np.uint8:
        gray_img = (gray_img * 255).astype(np.uint8)
    
    return entropy(gray_img, disk(radius))
```

**Use Cases:**
- Texture complexity at fine scale
- Detecting heterogeneous regions
- Identifying textured vs. smooth areas

**Computational Cost:** Moderate (~0.3s for 512×512 image)  
**Clinical Value:** ⭐⭐⭐⭐

---

## Additional Per-Block Methods

These compute features over small patches (e.g., 16×16 or 32×32), providing more stable estimates.

### 6. Fractal Dimension

**Concept:** Measure of surface complexity and self-similarity.

```python
def fractal_dimension_map(gray_img, patch_size=32, stride=8):
    """
    Compute fractal dimension for each patch using box-counting method.
    
    Higher values = more complex/irregular texture
    Lower values = smoother texture
    """
    H, W = gray_img.shape
    ys = range(0, H - patch_size + 1, stride)
    xs = range(0, W - patch_size + 1, stride)
    
    fd_map = np.zeros((len(ys), len(xs)))
    
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch = gray_img[y:y+patch_size, x:x+patch_size]
            fd_map[i, j] = compute_fractal_dimension(patch)
    
    return fd_map, (ys, xs)

def compute_fractal_dimension(patch, threshold=None):
    """
    Simplified box-counting fractal dimension.
    """
    if threshold is None:
        threshold = np.mean(patch)
    
    # Binarize
    Z = (patch > threshold)
    
    # Pad to power of 2
    p = int(np.ceil(np.log2(max(Z.shape))))
    n = 2**p
    Z_pad = np.zeros((n, n), dtype=bool)
    Z_pad[:Z.shape[0], :Z.shape[1]] = Z
    
    # Box counting
    counts = []
    box_sizes = 2**np.arange(p, 0, -1)
    
    for size in box_sizes:
        count = 0
        for i in range(0, n, size):
            for j in range(0, n, size):
                if Z_pad[i:i+size, j:j+size].any():
                    count += 1
        counts.append(count)
    
    # Fit log-log slope
    coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
    return -coeffs[0]  # Fractal dimension
```

**Use Cases:**
- Mucosal surface complexity
- Quantifying irregularity
- Detecting fine structural changes
- Research-level texture characterization

**Computational Cost:** Slow (~0.5-1s per patch)  
**Clinical Value:** ⭐⭐⭐

---

### 7. Histogram-Based Features (Per-Block)

**Concept:** Statistical measures derived from intensity histograms.

```python
def histogram_features_map(gray_img, patch_size=32, stride=8, bins=32):
    """
    Compute histogram-based features for each patch.
    
    Features:
    - Histogram spread (IQR)
    - Peak sharpness (kurtosis)
    - Number of modes (peaks)
    - Histogram uniformity
    """
    H, W = gray_img.shape
    ys = range(0, H - patch_size + 1, stride)
    xs = range(0, W - patch_size + 1, stride)
    
    feature_maps = {
        'iqr': np.zeros((len(ys), len(xs))),
        'kurtosis': np.zeros((len(ys), len(xs))),
        'uniformity': np.zeros((len(ys), len(xs))),
    }
    
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch = gray_img[y:y+patch_size, x:x+patch_size]
            
            # Compute histogram
            hist, _ = np.histogram(patch, bins=bins, range=(0, 256))
            hist = hist / hist.sum()  # Normalize
            
            # IQR
            cumsum = np.cumsum(hist)
            q75_idx = np.searchsorted(cumsum, 0.75)
            q25_idx = np.searchsorted(cumsum, 0.25)
            feature_maps['iqr'][i, j] = q75_idx - q25_idx
            
            # Kurtosis (using scipy)
            from scipy.stats import kurtosis
            feature_maps['kurtosis'][i, j] = kurtosis(hist)
            
            # Uniformity
            feature_maps['uniformity'][i, j] = np.sum(hist**2)
    
    return feature_maps, (ys, xs)
```

**Use Cases:**
- Distribution shape analysis
- Detecting multi-modal textures
- Complement to GLCM features

**Computational Cost:** Fast-Moderate (~0.01s per patch)  
**Clinical Value:** ⭐⭐⭐

---

## Frequency & Multi-Scale Methods

### 8. FFT-Based Texture Energy

**Concept:** Analyze texture in frequency domain to detect periodic patterns and directionality.

```python
from scipy.fft import fft2, fftshift

def fft_texture_map(gray_img, patch_size=32, stride=8):
    """
    Compute frequency-domain texture features per patch.
    
    Features extracted:
    - Spectral energy (overall frequency content)
    - Dominant frequency
    - Directionality (anisotropy)
    """
    H, W = gray_img.shape
    ys = range(0, H - patch_size + 1, stride)
    xs = range(0, W - patch_size + 1, stride)
    
    energy_map = np.zeros((len(ys), len(xs)))
    dominant_freq_map = np.zeros((len(ys), len(xs)))
    
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch = gray_img[y:y+patch_size, x:x+patch_size]
            
            # 2D FFT
            f_transform = fftshift(fft2(patch))
            power_spectrum = np.abs(f_transform)**2
            
            # Remove DC component (center)
            center = patch_size // 2
            power_spectrum[center-2:center+2, center-2:center+2] = 0
            
            # Spectral energy (log scale for visualization)
            energy_map[i, j] = np.log1p(np.sum(power_spectrum))
            
            # Dominant frequency (distance from center to max power)
            max_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
            dominant_freq_map[i, j] = np.sqrt((max_idx[0] - center)**2 + 
                                               (max_idx[1] - center)**2)
    
    return {'energy': energy_map, 'dominant_freq': dominant_freq_map}, (ys, xs)

def directional_fft_energy(patch, num_directions=8):
    """
    Compute directional frequency energy (texture orientation).
    """
    f_transform = fftshift(fft2(patch))
    power_spectrum = np.abs(f_transform)**2
    
    center = patch.shape[0] // 2
    y, x = np.ogrid[:patch.shape[0], :patch.shape[1]]
    y = y - center
    x = x - center
    
    angles = np.arctan2(y, x)
    
    # Divide spectrum into directional wedges
    dir_energies = []
    for i in range(num_directions):
        angle_min = -np.pi + i * (2*np.pi / num_directions)
        angle_max = angle_min + (2*np.pi / num_directions)
        mask = (angles >= angle_min) & (angles < angle_max)
        dir_energies.append(np.sum(power_spectrum[mask]))
    
    # Return dominant direction and its strength
    dominant_dir = np.argmax(dir_energies)
    directionality = (max(dir_energies) - min(dir_energies)) / (max(dir_energies) + 1e-10)
    
    return dominant_dir, directionality
```

**Use Cases:**
- Detecting periodic vascular patterns
- Mucosal fold orientation
- Assessing texture regularity
- Identifying directional bleeding patterns

**Computational Cost:** Moderate (~0.1s per patch)  
**Clinical Value:** ⭐⭐⭐

---

### 9. Wavelet Decomposition

**Concept:** Multi-resolution texture analysis using wavelet transform.

```python
import pywt

def wavelet_features_map(gray_img, patch_size=32, stride=8, wavelet='haar'):
    """
    Compute wavelet-based texture features per patch.
    
    Decomposes texture into:
    - Approximation (low-frequency content)
    - Horizontal detail (horizontal edges)
    - Vertical detail (vertical edges)
    - Diagonal detail (diagonal edges)
    """
    H, W = gray_img.shape
    ys = range(0, H - patch_size + 1, stride)
    xs = range(0, W - patch_size + 1, stride)
    
    feature_maps = {
        'approx_energy': np.zeros((len(ys), len(xs))),
        'horiz_energy': np.zeros((len(ys), len(xs))),
        'vert_energy': np.zeros((len(ys), len(xs))),
        'diag_energy': np.zeros((len(ys), len(xs))),
    }
    
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch = gray_img[y:y+patch_size, x:x+patch_size]
            
            # 2D Discrete Wavelet Transform
            coeffs = pywt.dwt2(patch, wavelet)
            cA, (cH, cV, cD) = coeffs
            
            # Compute energies
            feature_maps['approx_energy'][i, j] = np.sum(cA**2)
            feature_maps['horiz_energy'][i, j] = np.sum(cH**2)
            feature_maps['vert_energy'][i, j] = np.sum(cV**2)
            feature_maps['diag_energy'][i, j] = np.sum(cD**2)
    
    return feature_maps, (ys, xs)

def multiscale_wavelet_energy(patch, wavelet='db4', max_level=3):
    """
    Compute wavelet energy at multiple decomposition levels.
    
    Captures texture at different spatial scales.
    """
    energies = []
    
    for level in range(1, max_level + 1):
        coeffs = pywt.wavedec2(patch, wavelet, level=level)
        
        # Sum energy across all detail coefficients at this level
        detail_energy = sum(np.sum(c**2) for c in coeffs[1:])
        energies.append(detail_energy)
    
    return energies
```

**Use Cases:**
- Multi-scale texture characterization
- Edge detection at multiple resolutions
- Separating coarse vs. fine texture
- Texture orientation analysis

**Computational Cost:** Moderate (~0.1-0.2s per patch)  
**Clinical Value:** ⭐⭐⭐

**Note:** Requires `pywt` package: `pip install PyWavelets`

---

## Color-Based Methods

For RGB colonoscopy images (before grayscale conversion).

### 10. Color Channel Statistics

```python
def color_texture_maps(rgb_img, patch_size=32, stride=8):
    """
    Compute color-based texture features.
    
    Particularly useful for medical imaging:
    - Red channel: Inflammation, bleeding
    - Green channel: Healthy mucosa
    - Blue channel: Vessels, depth
    """
    H, W = rgb_img.shape[:2]
    ys = range(0, H - patch_size + 1, stride)
    xs = range(0, W - patch_size + 1, stride)
    
    feature_maps = {
        'red_mean': np.zeros((len(ys), len(xs))),
        'red_std': np.zeros((len(ys), len(xs))),
        'redness_index': np.zeros((len(ys), len(xs))),
        'color_variance': np.zeros((len(ys), len(xs))),
    }
    
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch = rgb_img[y:y+patch_size, x:x+patch_size]
            
            # Red channel statistics
            feature_maps['red_mean'][i, j] = np.mean(patch[:, :, 0])
            feature_maps['red_std'][i, j] = np.std(patch[:, :, 0])
            
            # Redness index: R / (R+G+B)
            total = np.sum(patch, axis=2) + 1e-10
            redness = patch[:, :, 0] / total
            feature_maps['redness_index'][i, j] = np.mean(redness)
            
            # Color variance (total color variation)
            feature_maps['color_variance'][i, j] = np.var(patch)
    
    return feature_maps, (ys, xs)
```

### 11. HSV Color Space Features

```python
def hsv_texture_maps(rgb_img, patch_size=32, stride=8):
    """
    Compute texture features in HSV color space.
    
    HSV is often more intuitive for medical imaging:
    - Hue: Color type (red=inflammation, yellow=bile, etc.)
    - Saturation: Color purity (vivid vs. pale)
    - Value: Brightness
    """
    from skimage.color import rgb2hsv
    
    hsv_img = rgb2hsv(rgb_img)
    H, W = hsv_img.shape[:2]
    ys = range(0, H - patch_size + 1, stride)
    xs = range(0, W - patch_size + 1, stride)
    
    feature_maps = {
        'hue_mean': np.zeros((len(ys), len(xs))),
        'hue_variance': np.zeros((len(ys), len(xs))),
        'saturation_mean': np.zeros((len(ys), len(xs))),
        'value_std': np.zeros((len(ys), len(xs))),
    }
    
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch = hsv_img[y:y+patch_size, x:x+patch_size]
            
            feature_maps['hue_mean'][i, j] = np.mean(patch[:, :, 0])
            feature_maps['hue_variance'][i, j] = np.var(patch[:, :, 0])
            feature_maps['saturation_mean'][i, j] = np.mean(patch[:, :, 1])
            feature_maps['value_std'][i, j] = np.std(patch[:, :, 2])
    
    return feature_maps, (ys, xs)

def color_coherence_map(rgb_img, patch_size=32, stride=8):
    """
    Measure color uniformity within patches.
    
    Low coherence = mixed colors (vascular patterns, inflammation)
    High coherence = uniform color (healthy mucosa)
    """
    from skimage.color import rgb2lab
    
    lab_img = rgb2lab(rgb_img)  # Perceptually uniform color space
    H, W = rgb_img.shape[:2]
    ys = range(0, H - patch_size + 1, stride)
    xs = range(0, W - patch_size + 1, stride)
    
    coherence_map = np.zeros((len(ys), len(xs)))
    
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch = lab_img[y:y+patch_size, x:x+patch_size]
            
            # Compute Euclidean distance in LAB space from patch mean
            mean_color = np.mean(patch.reshape(-1, 3), axis=0)
            distances = np.sqrt(np.sum((patch - mean_color)**2, axis=2))
            
            # Coherence = inverse of mean distance
            coherence_map[i, j] = 1.0 / (np.mean(distances) + 1e-10)
    
    return coherence_map, (ys, xs)
```

**Use Cases:**
- Inflammation detection (redness)
- Vascular pattern prominence
- Bleeding identification
- Bile staining detection
- Healthy vs. diseased tissue discrimination

**Computational Cost:** Fast-Moderate (~0.05-0.1s per patch)  
**Clinical Value:** ⭐⭐⭐⭐⭐ (if RGB available)

---

## Trade-offs & Performance

### Computational Complexity Comparison

| Method | Complexity per Pixel/Patch | Speed (512×512) | Memory | Interpretability |
|--------|---------------------------|-----------------|---------|------------------|
| **First-Order Stats** | O(N) | <0.01s | Low | ⭐⭐⭐⭐⭐ |
| **Local Std/Range** | O(N·k²) | ~0.1s | Low | ⭐⭐⭐⭐⭐ |
| **Gradient/Sobel** | O(N) | ~0.05s | Low | ⭐⭐⭐⭐ |
| **LBP** | O(N·P) | ~0.2s | Low | ⭐⭐⭐ |
| **GLCM** | O(N·L²) | ~5s (32×32 patches) | Medium | ⭐⭐⭐⭐ |
| **FFT** | O(N·log(N)) | ~0.5s (32×32 patches) | Medium | ⭐⭐ |
| **Wavelets** | O(N·log(N)) | ~0.5s (32×32 patches) | Medium | ⭐⭐⭐ |
| **Fractal Dimension** | O(N²·log(N)) | ~30s (32×32 patches) | Low | ⭐⭐ |
| **Color Features** | O(N) | ~0.1s | Medium | ⭐⭐⭐⭐ |

Where:
- N = number of pixels
- k = kernel size
- L = GLCM quantization levels
- P = number of LBP neighbors

### Spatial Resolution Comparison

| Approach | Resolution | Smoothness | Noise Robustness | Detail Preservation |
|----------|-----------|------------|------------------|-------------------|
| **Per-Pixel (3×3 kernel)** | Highest | Low | Low | ⭐⭐⭐⭐⭐ |
| **Per-Pixel (7×7 kernel)** | High | Medium | Medium | ⭐⭐⭐⭐ |
| **Small Blocks (8×8, stride=4)** | High | Medium | High | ⭐⭐⭐⭐ |
| **Medium Blocks (32×32, stride=8)** | ✅ **Recommended** | High | Very High | ⭐⭐⭐ |
| **Large Blocks (64×64, stride=16)** | Low | Very High | Very High | ⭐⭐ |

### Block Size Selection Guide

```python
# For different analysis goals:

# Fine-grained analysis (small lesions, detailed patterns)
WIN = 16
STEP = 4

# Balanced analysis (recommended for most cases) ✅
WIN = 32
STEP = 8

# Coarse-grained analysis (large regions, stable statistics)
WIN = 64
STEP = 16

# Super-resolution analysis (research, slow)
WIN = 8
STEP = 2
```

---

## Medical Imaging Recommendations

### Priority Ranking for Mayo Score Discrimination

Based on clinical relevance and computational efficiency:

#### **Tier 1: Essential (Implement First)** ⭐⭐⭐⭐⭐

1. **Local Standard Deviation** (Per-Pixel)
   - **Why:** Directly captures mucosal roughness
   - **Mayo Correlation:** Strong (M0 smooth → M3 rough)
   - **Speed:** Fast
   - **Implementation:** Easy

2. **GLCM Contrast** (Per-Block) [Already Implemented ✅]
   - **Why:** Gold standard for texture roughness
   - **Mayo Correlation:** Very strong
   - **Speed:** Moderate
   - **Proven:** Extensive medical imaging literature

3. **GLCM Homogeneity** (Per-Block) [Already Implemented ✅]
   - **Why:** Inverse of roughness, captures smoothness
   - **Mayo Correlation:** Very strong (inverse)
   - **Complementary:** Pairs well with contrast

4. **Entropy** (First-Order) [Already Implemented ✅]
   - **Why:** Measures tissue disorder/complexity
   - **Mayo Correlation:** Strong
   - **Speed:** Very fast

#### **Tier 2: Highly Recommended** ⭐⭐⭐⭐

5. **Edge Density Map**
   - **Why:** Vascular patterns, ulceration edges
   - **Mayo Correlation:** Moderate-Strong
   - **Speed:** Fast
   - **Added Value:** Captures different aspect than GLCM

6. **Local Binary Patterns**
   - **Why:** Micro-texture changes, rotation-invariant
   - **Mayo Correlation:** Moderate
   - **Speed:** Fast
   - **Robustness:** Very stable

7. **Red Channel Intensity** (If RGB Available)
   - **Why:** Direct inflammation indicator
   - **Mayo Correlation:** Strong (inflammation = redness)
   - **Speed:** Very fast
   - **Clinical:** Intuitive interpretation

8. **Color Coherence** (If RGB Available)
   - **Why:** Vascular pattern complexity
   - **Mayo Correlation:** Moderate
   - **Uniqueness:** Not captured by grayscale

#### **Tier 3: Research/Optional** ⭐⭐⭐

9. **Wavelet Energy**
   - **Why:** Multi-scale analysis
   - **Mayo Correlation:** Unknown (requires validation)
   - **Speed:** Moderate
   - **Research Value:** High

10. **FFT Directionality**
    - **Why:** Mucosal fold orientation
    - **Mayo Correlation:** Moderate
    - **Speed:** Moderate
    - **Specificity:** Captures unique patterns

11. **Fractal Dimension**
    - **Why:** Surface complexity quantification
    - **Mayo Correlation:** Likely strong
    - **Speed:** Slow
    - **Research:** Novel application

### Multi-Feature Fusion Strategy

For best discriminative power, combine complementary features:

```python
# Recommended feature combination (3-5 features):

RECOMMENDED_FEATURES = [
    'Entropy',              # Disorder (fast, first-order)
    'Contrast',             # Roughness (GLCM)
    'Homogeneity',          # Smoothness (GLCM)
    'Local_Std',            # Per-pixel roughness (new)
    'Edge_Density',         # Vascular/boundary (new)
]

# With RGB available, add:
RGB_FEATURES = [
    'Redness_Index',        # Inflammation
    'Color_Coherence',      # Uniformity
]

# For research/publication:
ADVANCED_FEATURES = [
    'LBP_Variance',         # Micro-texture
    'Wavelet_Diagonal',     # Multi-scale edges
    'Fractal_Dimension',    # Complexity measure
]
```

### Optimal Configuration for Mayo Analysis

```python
# Recommended parameters based on colonoscopy image characteristics:

# Window/Block size
WIN = 32                    # Good balance: stable statistics, sufficient detail
STEP = 8                    # 75% overlap for smooth heatmaps

# GLCM parameters
GLCM_LEVELS = 32           # Sufficient discrimination, not too slow
OFFSETS = [(1,0), (1,-1), (0,-1), (-1,-1)]  # 4 directions for rotation invariance
DISTANCES = [1, 2]         # Short distances for local texture

# LBP parameters
LBP_RADIUS = 1             # Captures immediate neighbors
LBP_POINTS = 8             # Standard configuration
LBP_METHOD = 'uniform'     # Rotation-invariant

# Local operators (per-pixel)
LOCAL_KERNEL = 7           # Odd number, captures ~7×7 neighborhood
GRADIENT_THRESHOLD = 75    # Percentile for edge detection

# Color (if available)
COLOR_SPACE = 'HSV'        # More intuitive than RGB for medical
USE_LAB_FOR_COHERENCE = True  # Perceptually uniform
```

---

## Implementation Examples

### Example 1: Adding Local Standard Deviation to Your Notebook

```python
# Add this function to your existing feature extraction code:

def compute_local_std_feature_map(gray_img_u8, kernel_size=7):
    """
    Generate per-pixel local standard deviation heatmap.
    
    This is FASTER than block-based GLCM and provides smooth heatmaps.
    """
    from scipy.ndimage import uniform_filter
    
    # Convert to float
    img = gray_img_u8.astype(np.float32)
    
    # Compute local mean and mean of squares
    mean = uniform_filter(img, kernel_size)
    mean_sq = uniform_filter(img**2, kernel_size)
    
    # Variance = E[X²] - E[X]²
    variance = mean_sq - mean**2
    variance = np.maximum(variance, 0)  # Ensure non-negative
    
    # Standard deviation
    std_map = np.sqrt(variance)
    
    return std_map

# Usage with your existing overlay code:
def plot_local_std_overlay(gray_img_u8, image_name="", save_dir=None):
    """
    Create and save local std heatmap overlay.
    """
    # Compute feature map
    std_map = compute_local_std_feature_map(gray_img_u8, kernel_size=7)
    
    # Normalize for visualization
    std_map_norm = (std_map - std_map.min()) / (std_map.max() - std_map.min() + 1e-10)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(gray_img_u8, cmap='gray')
    axes[0].set_title(f'{image_name} - Original')
    axes[0].axis('off')
    
    # Heatmap only
    axes[1].imshow(std_map_norm, cmap='jet', interpolation='nearest')
    axes[1].set_title('Local Texture Roughness')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(gray_img_u8, cmap='gray')
    axes[2].imshow(std_map_norm, cmap='jet', alpha=0.45, interpolation='nearest')
    axes[2].set_title('Roughness Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{image_name}_LocalStd_overlay.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")
    
    plt.show()

# Run for all M0-M3 images:
for img_name in ["M0", "M1", "M2", "M3"]:
    img_path = find_image_path(IMG_DIR, img_name)
    img = load_image_for_texture(img_path)
    img_u8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    plot_local_std_overlay(img_u8, image_name=img_name, save_dir=OUT_DIR)
```

### Example 2: Edge Density Heatmap

```python
from scipy.ndimage import sobel, uniform_filter

def compute_edge_density_map(gray_img_u8, block_size=16, threshold_percentile=75):
    """
    Create edge density heatmap.
    
    Useful for detecting vascular patterns and mucosal complexity.
    """
    # Compute gradient magnitude
    grad_x = sobel(gray_img_u8.astype(float), axis=1)
    grad_y = sobel(gray_img_u8.astype(float), axis=0)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold to binary edge map
    threshold = np.percentile(gradient_mag, threshold_percentile)
    edges = (gradient_mag > threshold).astype(float)
    
    # Compute local edge density
    edge_density = uniform_filter(edges, size=block_size)
    
    return edge_density

def plot_edge_density_overlay(gray_img_u8, image_name="", save_dir=None):
    """
    Create edge density visualization.
    """
    # Compute feature map
    edge_map = compute_edge_density_map(gray_img_u8, block_size=16)
    
    # Normalize
    edge_map_norm = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-10)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(gray_img_u8, cmap='gray')
    axes[0].set_title(f'{image_name} - Original')
    axes[0].axis('off')
    
    axes[1].imshow(edge_map_norm, cmap='hot', interpolation='nearest')
    axes[1].set_title('Edge Density (Vascular Patterns)')
    axes[1].axis('off')
    
    axes[2].imshow(gray_img_u8, cmap='gray')
    axes[2].imshow(edge_map_norm, cmap='hot', alpha=0.45, interpolation='nearest')
    axes[2].set_title('Edge Density Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{image_name}_EdgeDensity_overlay.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")
    
    plt.show()
```

### Example 3: Multi-Feature Comparison Grid

```python
def plot_multifeature_comparison(gray_img_u8, image_name="M0", save_dir=None):
    """
    Show multiple feature maps in a grid for comparison.
    
    Displays 6 features:
    - Local Std
    - Edge Density
    - Entropy (block-based)
    - GLCM Contrast (block-based)
    - GLCM Homogeneity (block-based)
    - LBP Variance
    """
    from skimage.feature import local_binary_pattern
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    # Original
    axes[0].imshow(gray_img_u8, cmap='gray')
    axes[0].set_title(f'{image_name} - Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Feature 1: Local Std
    local_std = compute_local_std_feature_map(gray_img_u8, kernel_size=7)
    local_std_norm = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-10)
    im1 = axes[1].imshow(local_std_norm, cmap='jet', interpolation='nearest')
    axes[1].set_title('Local Std (Roughness)', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Feature 2: Edge Density
    edge_density = compute_edge_density_map(gray_img_u8, block_size=16)
    edge_density_norm = (edge_density - edge_density.min()) / (edge_density.max() - edge_density.min() + 1e-10)
    im2 = axes[2].imshow(edge_density_norm, cmap='hot', interpolation='nearest')
    axes[2].set_title('Edge Density', fontsize=12)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Feature 3: LBP Variance
    lbp_var = local_binary_pattern(gray_img_u8, P=8, R=1, method='var')
    lbp_var_norm = (lbp_var - lbp_var.min()) / (lbp_var.max() - lbp_var.min() + 1e-10)
    im3 = axes[3].imshow(lbp_var_norm, cmap='viridis', interpolation='nearest')
    axes[3].set_title('LBP Variance', fontsize=12)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    # Features 4-6: Block-based (use your existing functions)
    # Assuming you have these from your current implementation
    fmap_dict, meta = texture_feature_maps(
        gray_img_u8, win=32, step=8, levels=32,
        features_to_map=['Entropy', 'Contrast', 'Homogeneity']
    )
    
    # Feature 4: Entropy
    entropy_map = fmap_dict['Entropy']
    entropy_norm = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-10)
    im4 = axes[4].imshow(entropy_norm, cmap='plasma', interpolation='nearest')
    axes[4].set_title('Entropy (Block)', fontsize=12)
    axes[4].axis('off')
    plt.colorbar(im4, ax=axes[4], fraction=0.046)
    
    # Feature 5: GLCM Contrast
    contrast_map = fmap_dict['Contrast']
    contrast_norm = (contrast_map - contrast_map.min()) / (contrast_map.max() - contrast_map.min() + 1e-10)
    im5 = axes[5].imshow(contrast_norm, cmap='jet', interpolation='nearest')
    axes[5].set_title('GLCM Contrast', fontsize=12)
    axes[5].axis('off')
    plt.colorbar(im5, ax=axes[5], fraction=0.046)
    
    # Feature 6: GLCM Homogeneity
    homog_map = fmap_dict['Homogeneity']
    homog_norm = (homog_map - homog_map.min()) / (homog_map.max() - homog_map.min() + 1e-10)
    im6 = axes[6].imshow(homog_norm, cmap='jet_r', interpolation='nearest')  # Inverted colormap
    axes[6].set_title('GLCM Homogeneity', fontsize=12)
    axes[6].axis('off')
    plt.colorbar(im6, ax=axes[6], fraction=0.046)
    
    # Feature 7: Composite overlay (average of normalized features)
    composite = (local_std_norm + edge_density_norm + entropy_norm + contrast_norm) / 4
    axes[7].imshow(gray_img_u8, cmap='gray')
    axes[7].imshow(composite, cmap='jet', alpha=0.5, interpolation='nearest')
    axes[7].set_title('Composite Feature Overlay', fontsize=12, fontweight='bold')
    axes[7].axis('off')
    
    plt.suptitle(f'Multi-Feature Texture Analysis: {image_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{image_name}_MultiFeature_Comparison.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {out_path}")
    
    plt.show()

# Generate for all Mayo images
for img_name in ["M0", "M1", "M2", "M3"]:
    img_path = find_image_path(IMG_DIR, img_name)
    img = load_image_for_texture(img_path)
    img_u8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    plot_multifeature_comparison(img_u8, image_name=img_name, save_dir=OUT_DIR)
```

---

## Best Practices

### 1. Normalization & Preprocessing

```python
# Always normalize feature maps before visualization
def normalize_feature_map(fmap):
    """
    Normalize to [0, 1] range, handling NaN and infinities.
    """
    fmap = np.asarray(fmap, dtype=np.float32)
    
    # Remove non-finite values
    finite_mask = np.isfinite(fmap)
    if not np.any(finite_mask):
        return np.zeros_like(fmap)
    
    # Get min/max from finite values only
    fmin = np.min(fmap[finite_mask])
    fmax = np.max(fmap[finite_mask])
    
    if fmax - fmin < 1e-12:
        return np.zeros_like(fmap)
    
    # Normalize
    fmap_norm = (fmap - fmin) / (fmax - fmin)
    fmap_norm[~finite_mask] = 0  # Set non-finite to 0
    
    return fmap_norm

# Percentile-based normalization (robust to outliers)
def normalize_robust(fmap, lower_percentile=2, upper_percentile=98):
    """
    Robust normalization using percentiles instead of min/max.
    """
    vmin = np.percentile(fmap[np.isfinite(fmap)], lower_percentile)
    vmax = np.percentile(fmap[np.isfinite(fmap)], upper_percentile)
    
    fmap_norm = np.clip((fmap - vmin) / (vmax - vmin + 1e-10), 0, 1)
    return fmap_norm
```

### 2. Color Map Selection

```python
# Recommended colormaps for different feature types:

COLORMAP_GUIDE = {
    # For features where high = problematic (roughness, complexity)
    'roughness': 'jet',        # Blue (low/good) → Red (high/bad)
    'complexity': 'hot',       # Black → Red → Yellow (intensity scale)
    'disorder': 'plasma',      # Purple → Yellow (perceptually uniform)
    
    # For features where high = healthy (smoothness, uniformity)
    'smoothness': 'jet_r',     # Inverted: Red (low/bad) → Blue (high/good)
    'uniformity': 'YlGn',      # Yellow → Green (healthy indicator)
    
    # For neutral features (informational)
    'general': 'viridis',      # Blue → Yellow (perceptually uniform)
    'edges': 'binary',         # Black & white (for binary features)
    
    # For medical overlay (visibility on gray background)
    'medical_overlay': 'RdYlBu_r',  # Red-Yellow-Blue (intuitive for inflammation)
}

# Usage:
plt.imshow(roughness_map, cmap='jet', alpha=0.45)
plt.imshow(smoothness_map, cmap='jet_r', alpha=0.45)
```

### 3. Consistent Color Scaling Across Images

```python
def plot_feature_with_shared_scale(feature_maps_dict, feature_name, 
                                   image_names=["M0", "M1", "M2", "M3"]):
    """
    Plot feature maps with shared color scale for fair comparison.
    
    Critical for comparing Mayo scores!
    """
    # Find global min/max across ALL images
    all_values = np.concatenate([feature_maps_dict[name][feature_name].ravel() 
                                  for name in image_names])
    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)
    
    if vmax - vmin < 1e-12:
        vmin, vmax = 0, 1
    
    # Plot all images with same scale
    fig, axes = plt.subplots(1, len(image_names), figsize=(20, 5))
    
    for i, img_name in enumerate(image_names):
        fmap = feature_maps_dict[img_name][feature_name]
        
        im = axes[i].imshow(fmap, cmap='jet', vmin=vmin, vmax=vmax, 
                            interpolation='nearest')
        axes[i].set_title(f'{img_name}\n{feature_name}', fontsize=12)
        axes[i].axis('off')
    
    # Single colorbar for all subplots
    fig.colorbar(im, ax=axes, orientation='horizontal', 
                 fraction=0.05, pad=0.05, label=feature_name)
    
    plt.tight_layout()
    plt.show()
```

### 4. Overlay Transparency Guidelines

```python
# Recommended alpha (transparency) values:

ALPHA_GUIDELINES = {
    'subtle_overlay': 0.3,      # For dense/busy features (LBP, high-res maps)
    'standard_overlay': 0.45,   # ✅ Recommended default
    'prominent_overlay': 0.6,   # For sparse features (edge maps)
    'diagnostic': 0.7,          # When feature is primary focus
}

# Adaptive alpha based on feature density
def adaptive_alpha(feature_map, base_alpha=0.45):
    """
    Adjust transparency based on feature map density.
    """
    density = np.mean(feature_map > np.mean(feature_map))
    
    if density > 0.7:  # Very dense
        return base_alpha * 0.7
    elif density < 0.3:  # Very sparse
        return base_alpha * 1.3
    else:
        return base_alpha
```

### 5. Multi-Resolution Analysis

```python
def multiscale_feature_analysis(gray_img_u8, feature_func, scales=[0.5, 1.0, 2.0]):
    """
    Compute features at multiple image scales.
    
    Useful for detecting texture at different spatial frequencies.
    """
    from skimage.transform import rescale
    
    results = {}
    
    for scale in scales:
        # Rescale image
        if scale != 1.0:
            img_scaled = rescale(gray_img_u8, scale, preserve_range=True, 
                                 anti_aliasing=True).astype(np.uint8)
        else:
            img_scaled = gray_img_u8
        
        # Compute feature
        feature_map = feature_func(img_scaled)
        
        # Resize back to original dimensions
        if scale != 1.0:
            from skimage.transform import resize
            feature_map = resize(feature_map, gray_img_u8.shape, 
                                preserve_range=True, anti_aliasing=True)
        
        results[f'scale_{scale}'] = feature_map
    
    return results

# Example usage:
scales = multiscale_feature_analysis(
    gray_img_u8, 
    lambda img: compute_local_std_feature_map(img, kernel_size=7),
    scales=[0.5, 1.0, 1.5]
)

# Combine scales (e.g., average)
combined = np.mean([scales[k] for k in scales], axis=0)
```

### 6. Statistical Validation

```python
def correlate_feature_with_mayo(feature_maps_dict, feature_name, 
                                mayo_scores={"M0": 0, "M1": 1, "M2": 2, "M3": 3}):
    """
    Compute correlation between feature and Mayo score.
    
    Helps identify which features best discriminate disease severity.
    """
    from scipy.stats import spearmanr, pearsonr
    
    mayo_vals = []
    feature_vals = []
    
    for img_name in sorted(mayo_scores.keys()):
        mayo_vals.append(mayo_scores[img_name])
        
        # Use mean feature value across image
        fmap = feature_maps_dict[img_name][feature_name]
        feature_vals.append(np.nanmean(fmap))
    
    # Compute correlations
    pearson_r, pearson_p = pearsonr(mayo_vals, feature_vals)
    spearman_r, spearman_p = spearmanr(mayo_vals, feature_vals)
    
    print(f"\n{feature_name} vs Mayo Score:")
    print(f"  Pearson:  r={pearson_r:.3f}, p={pearson_p:.4f}")
    print(f"  Spearman: ρ={spearman_r:.3f}, p={spearman_p:.4f}")
    
    # Interpretation
    if abs(spearman_r) > 0.8:
        print(f"  → Very strong correlation ⭐⭐⭐⭐⭐")
    elif abs(spearman_r) > 0.6:
        print(f"  → Strong correlation ⭐⭐⭐⭐")
    elif abs(spearman_r) > 0.4:
        print(f"  → Moderate correlation ⭐⭐⭐")
    else:
        print(f"  → Weak correlation ⭐⭐")
    
    return {'pearson': (pearson_r, pearson_p), 
            'spearman': (spearman_r, spearman_p)}

# Run for all features
for feat in ['Entropy', 'Contrast', 'Homogeneity', 'Local_Std']:
    correlate_feature_with_mayo(all_feature_maps, feat)
```

### 7. Quality Control

```python
def validate_feature_map(feature_map, feature_name="Feature"):
    """
    Check feature map for common issues.
    """
    issues = []
    
    # Check for NaN
    if np.any(np.isnan(feature_map)):
        nan_pct = 100 * np.mean(np.isnan(feature_map))
        issues.append(f"Contains {nan_pct:.1f}% NaN values")
    
    # Check for Inf
    if np.any(np.isinf(feature_map)):
        inf_pct = 100 * np.mean(np.isinf(feature_map))
        issues.append(f"Contains {inf_pct:.1f}% infinite values")
    
    # Check for constant values
    if len(np.unique(feature_map[np.isfinite(feature_map)])) < 5:
        issues.append("Nearly constant (< 5 unique values)")
    
    # Check range
    vmin, vmax = np.nanmin(feature_map), np.nanmax(feature_map)
    if vmax - vmin < 1e-10:
        issues.append("Zero variance")
    
    # Report
    if issues:
        print(f"⚠️ {feature_name} validation warnings:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"✅ {feature_name} passed validation")
    
    return len(issues) == 0
```

---

## Summary & Quick Reference

### Feature Selection Flowchart

```
Start: What do you want to detect?
│
├─ Surface Roughness
│  ├─ Fast & simple → Local Std ⭐⭐⭐⭐⭐
│  └─ Detailed & proven → GLCM Contrast ⭐⭐⭐⭐⭐
│
├─ Vascular Patterns
│  ├─ Edge-based → Edge Density ⭐⭐⭐⭐
│  └─ Color-based → Redness Index ⭐⭐⭐⭐⭐
│
├─ Micro-texture Patterns
│  ├─ Rotation-invariant → LBP ⭐⭐⭐⭐
│  └─ Directional → Wavelet/FFT ⭐⭐⭐
│
├─ Overall Complexity
│  ├─ Fast → Entropy ⭐⭐⭐⭐⭐
│  └─ Research → Fractal Dimension ⭐⭐⭐
│
└─ Inflammation
   ├─ Direct → Red Channel / Redness Index ⭐⭐⭐⭐⭐
   └─ Indirect → Color Coherence ⭐⭐⭐⭐
```

### Recommended Minimal Implementation (3 Features)

For fastest implementation with high discriminative power:

1. **Local Standard Deviation** (Per-pixel, fast)
2. **GLCM Contrast** (Block-based, your current implementation)
3. **Entropy** (First-order, very fast, your current implementation)

### Recommended Standard Implementation (5 Features)

Balanced coverage of texture aspects:

1. **Local Standard Deviation** (Roughness, per-pixel)
2. **Edge Density** (Vascular patterns, per-pixel)
3. **GLCM Contrast** (Spatial roughness, block)
4. **GLCM Homogeneity** (Smoothness, block)
5. **Entropy** (Disorder, block)

### Recommended Advanced Implementation (8+ Features)

For research/publication:

1. All standard features above
2. **LBP Variance** (Micro-texture)
3. **Redness Index** (Inflammation, if RGB)
4. **Color Coherence** (Uniformity, if RGB)
5. **Wavelet Diagonal Energy** (Multi-scale)
6. **FFT Directionality** (Optional)
7. **Fractal Dimension** (Optional, slow)

---

## Next Steps

### Immediate Actions

1. ✅ **Keep your current implementation** (GLCM + First-order features)
   - It's already excellent and proven

2. 🆕 **Add Local Standard Deviation**
   - Fastest ROI (return on investment)
   - ~20 lines of code
   - Immediate visual impact

3. 🆕 **Add Edge Density Map**
   - Complementary to existing features
   - ~30 lines of code
   - Captures vascular patterns

4. 📊 **Generate Multi-Feature Comparison**
   - Use Example 3 above
   - Compare all features side-by-side for M0-M3
   - Identify which features show strongest Mayo discrimination

### Validation Steps

1. **Visual inspection:** Do heatmaps make clinical sense?
2. **Correlation analysis:** Compute Mayo vs. feature correlations
3. **Cross-validation:** Test on independent image set
4. **Statistical testing:** ANOVA or Kruskal-Wallis across Mayo groups

### For Publication

1. Select 3-5 top features based on correlation + interpretability
2. Create manuscript figures with consistent scales
3. Report quantitative metrics (mean ± std per Mayo score)
4. Include statistical significance tests
5. Discuss clinical relevance

---

## Conclusion

Your current implementation is already sophisticated and well-designed. The sliding-window GLCM approach is the gold standard in medical texture analysis.

**Key Recommendations:**
- ✅ **Your current features are excellent** - keep them
- 🎯 **Add Local Std and Edge Density** - highest impact, minimal effort
- 🎨 **Use consistent color scales** - critical for Mayo comparison
- 📊 **Quantify feature-Mayo correlations** - validate discriminative power
- 🔬 **Consider LBP for research** - adds novel micro-texture dimension

The balance between **computational speed**, **interpretability**, and **discriminative power** favors:
1. Local Standard Deviation (per-pixel)
2. GLCM Contrast (block-based)
3. Entropy (first-order)

These three features are fast, intuitive, clinically relevant, and provide complementary information about tissue texture.

---

**Document Version:** 1.0  
**Last Updated:** February 9, 2026  
**Author:** Texture Analysis Framework Discussion
