import numpy as np

import matplotlib.pyplot as plt

import os

from PIL import Image

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from skimage.feature import greycomatrix, greycoprops
from multiprocessing import Pool

from scipy import stats
from skimage import io

# Loading the images data and the respective labels data into images & labels variables

def loadImages():
    images, labels, labelNumbered = [], [], []

    mIndex = 1
    mMainDir = "Crops"

    mCropLabels = os.listdir(mMainDir)
    
    for mCrop in mCropLabels:
        if mCrop.startswith("."):
            d = 0
            mIndex = mIndex - 1
        else:
            for mImage in os.listdir(mMainDir + "/" + mCrop):
                if mImage.startswith("."):
                    s = 0
                else:
                    mImage = np.load(mMainDir + "/" + mCrop + "/" + mImage)
                    images.append(mImage.flatten() )

                    labels.append(mCrop)
                    labelNumbered.append(mIndex)
        mIndex = mIndex + 1
    
    return images, labels, labelNumbered



def glcm_props(patch):
    lf = []
    props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']

    # left nearest neighbor
    glcm = greycomatrix(patch, [1], [0], 256, symmetric=True, normed=True)
    for f in props:
        lf.append( greycoprops(glcm, f)[0,0] )

    # upper nearest neighbor
    glcm = greycomatrix(patch, [1], [np.pi/2], 256, symmetric=True, normed=True)
    for f in props:
        lf.append( greycoprops(glcm, f)[0,0] )
        
    return lf

def patch_gen(img, PAD=4):
    img1 = (img * 255).astype(np.uint8)

    W = img.shape[0]
    imgx = np.zeros((101+PAD*2, 101+PAD*2), dtype=img1.dtype)
    imgx[PAD:W+PAD,PAD:W+PAD] = img1
    imgx[:PAD,  PAD:W+PAD] = img1[PAD:0:-1,:]
    imgx[-PAD:, PAD:W+PAD] = img1[W-1:-PAD-1:-1,:]
    imgx[:, :PAD ] = imgx[:, PAD*2:PAD:-1]
    imgx[:, -PAD:] = imgx[:, W+PAD-1:-PAD*2-1:-1]

    xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, W))
    xx, yy = xx.flatten() + PAD, yy.flatten() + PAD

    for x, y in zip(xx, yy):
        patch = imgx[y-PAD:y+PAD+1, x-PAD:x+PAD+1]
        yield patch
        
def glcm_feature(img, verbose=False):
    
    H = 18
    W, NF, PAD = 101, 10, 4

    if img.sum() == 0:
        return np.zeros((W,H,NF), dtype=np.float32)
    
    l = []
    with Pool(3) as pool:
        for p in tqdm.tqdm(pool.imap(glcm_props, patch_gen(img, PAD)), total=W*H, disable=not verbose):
            l.append(p)
        
    fimg = np.array(l, dtype=np.float32).reshape(W, H, -1)
    return fimg

def visualize_glcm():
    img = red_band
    mask = red_band # read_mask(imgid)
    
    fimg = glcm_feature(img, verbose=1)
    
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(6,3))
    ax0.imshow(img)
    ax1.imshow(mask)
    plt.show()
    
    amin = np.amin(fimg, axis=(0,1))
    amax = np.amax(fimg, axis=(0,1))
    fimg = (fimg - amin) / (amax - amin)

    fimg[...,4] = np.power(fimg[...,4], 3)
    fimg[...,9] = np.power(fimg[...,9], 3)

    _, axs = plt.subplots(2, 5, figsize=(15,6))
    axs = axs.flatten()

    for k in range(fimg.shape[-1]):
        axs[k].imshow(fimg[...,k])
    plt.show()
    
def offset(length, angle):
    """Return the offset in pixels for a given length and angle"""
    dv = length * np.sign(-np.sin(angle)).astype(np.int32)
    dh = length * np.sign(np.cos(angle)).astype(np.int32)
    return dv, dh

def crop(img, center, win):
    """Return a square crop of img centered at center (side = 2*win + 1)"""
    row, col = center
    side = 2*win + 1
    first_row = row - win
    first_col = col - win
    last_row = first_row + side    
    last_col = first_col + side
    return img[first_row: last_row, first_col: last_col]

def cooc_maps(img, center, win, d=[1], theta=[0], levels=256):
    """
    Return a set of co-occurrence maps for different d and theta in a square 
    crop centered at center (side = 2*w + 1)
    """
    shape = (2*win + 1, 2*win + 1, len(d), len(theta))
    cooc = np.zeros(shape=shape, dtype=np.int32)
    row, col = center
    Ii = crop(img, (row, col), win)
    for d_index, length in enumerate(d):
        for a_index, angle in enumerate(theta):
            dv, dh = offset(length, angle)
            Ij = crop(img, center=(row + dv, col + dh), win=win)
            cooc[:, :, d_index, a_index] = encode_cooccurrence(Ii, Ij, levels)
    return cooc

def encode_cooccurrence(x, y, levels=256):
    """Return the code corresponding to co-occurrence of intensities x and y"""
    return x*levels + y

def decode_cooccurrence(code, levels=256):
    """Return the intensities x, y corresponding to code"""
    return code//levels, np.mod(code, levels)    

def compute_glcms(cooccurrence_maps, levels=256):
    """Compute the cooccurrence frequencies of the cooccurrence maps"""
    Nr, Na = cooccurrence_maps.shape[2:]
    glcms = np.zeros(shape=(levels, levels, Nr, Na), dtype=np.float64)
    for r in range(Nr):
        for a in range(Na):
            table = stats.itemfreq(cooccurrence_maps[:, :, r, a])
            codes = table[:, 0]
            freqs = table[:, 1]/float(table[:, 1].sum())
            i, j = decode_cooccurrence(codes, levels=levels)
            glcms[i, j, r, a] = freqs
    return glcms

def compute_props(glcms, props=('contrast',)):
    """Return a feature vector corresponding to a set of GLCM"""
    Nr, Na = glcms.shape[2:]
    features = np.zeros(shape=(Nr, Na, len(props)))
    for index, prop_name in enumerate(props):
        features[:, :, index] = greycoprops(glcms, prop_name)
    return features.ravel()

def haralick_features(img, win, d, theta, levels, props):
    """Return a map of Haralick features (one feature vector per pixel)"""
    rows, cols = img.shape
    margin = win + max(d)
    arr = np.pad(img, margin, mode='reflect')
    n_features = len(d) * len(theta) * len(props)
    feature_map = np.zeros(shape=(rows, cols, n_features), dtype=np.float64)
    for m in range(rows):
        for n in range(cols):
            coocs = cooc_maps(arr, (m + margin, n + margin), win, d, theta, levels)
            glcms = compute_glcms(coocs, levels)
            feature_map[m, n, :] = compute_props(glcms, props)
    return feature_map

# Saving generated textural features 

def save_labelled_train_data (data, mFieldIndex):
    mainDir = "Textures/Train"
    np.save(mainDir + '/field_texture_' + str(mFieldIndex) + '.npy', data )
    
def save_labelled_test_data (data, mFieldIndex):
    mainDir = "Textures/Test"
    np.save(mainDir + '/field_texture_' + str(mFieldIndex) + '.npy', data )

# Loading generated textural features 
    
def load_labelled_train_data (mFieldIndex):
    mainDir = "Textures/Train"
    mFieldImage = np.load(mainDir + '/field_tesxture_' + str(mFieldIndex) + '.npy')

def load_labelled_test_data (mFieldIndex):
    mainDir = "Textures/Test"
    mFieldImage = np.load(mainDir + '/field_tesxture_' + str(mFieldIndex) + '.npy')

# Splitting the data set into training and test data

X_train, X_test, y_train, y_test  = train_test_split(images, labelNumbered, train_size=0.7,test_size=0.3, random_state=42) 

# Distance

d = (1, 2)

# Angle of textural featuress

theta = (0 , np.pi/4, np.pi/2, 3*np.pi/4)

# Texture features to be generated

props = ('contrast', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 'correlation')

# Number of gray levels

levels = 256

# Windows Size

win = 19

for mCount in np.arange(0, len(X_train)):
    %time feature_map = haralick_features(X_train[mCount].reshape(100, 100), win, d, theta, levels, props)

    save_labelled_train_data(feature_map.reshape(100, 100, 48), mCount)

for mCount in np.arange(0, len(X_test)):
    %time feature_map = haralick_features(X_test[mCount].reshape(100, 100), win, d, theta, levels, props)
        
    save_labelled_test_data(feature_map.reshape(100, 100, 48), mCount)