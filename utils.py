import cv2
import time
from error_diffusion import error_diffusion
import numpy as np

class clock():
    def tick(self):
        self.st = time.time()
    
    def tock(self):
        self.et = time.time()
        
    def get_time(self):
        return self.et - self.st

def read_image(path, resize=None, binary=False):
    if not path: return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if resize: img = cv2.resize(img, (resize[1], resize[0]))
    img = img / 255.
    if binary:
        # img[img > 0.5] = 1.0
        # img[img <= 0.5] = 0.0
        # std_dev = sigma(img, filter_size=9)
        # ref = np.zeros(std_dev.shape)
        # ref[np.abs(std_dev)<0.03] = 255
        # cv2.imwrite('ref.png', ref)
        
        img = error_diffusion(img*255)
        ref = np.ones(img.shape) * 255
        img = median_filter(img, ref, filter_size=3)
        cv2.imwrite('halftone_secret.png', img*255)
        raise
    return img

def save_image(path, img):
    if np.max(img) <= 1.0: img = img * 255.
    cv2.imwrite(path, img)

def show_image(img, title=None):
    plt.figure()
    if title: plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()

def decode_shares(share1, share2):
    # We use XNOR for decoding rather than simple adding
    # We would the
    decode = share1 + share2
    decode[decode==1] = 0
    decode[decode==2] = 1
    # decode = np.logical_not(np.logical_xor(share1, share2)).astype(float)
    return decode

def median_filter(img, ref, filter_size=3):
    img_padded = np.pad(img,  [(int((filter_size-1)/2),int((filter_size-1)/2)),(int((filter_size-1)/2),int((filter_size-1)/2))], 'symmetric')
    output = img
    y_start, y_end = int((filter_size-1)/2), int(img.shape[0]+(filter_size-1)/2)
    x_start, x_end = int((filter_size-1)/2), int(img.shape[1]+(filter_size-1)/2)
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            if ref[i-y_start, j-x_start] == 255:
                output[i-y_start, j-x_start] = np.median(img_padded[i-y_start:i+y_start+1, j-y_start:j+y_start+1])
    return output

def sigma(img, filter_size=3):
    img_padded = np.pad(img,  [(int((filter_size-1)/2),int((filter_size-1)/2)),(int((filter_size-1)/2),int((filter_size-1)/2))], 'symmetric')
    output = np.zeros(img.shape)
    y_start, y_end = int((filter_size-1)/2), int(img.shape[0]+(filter_size-1)/2)
    x_start, x_end = int((filter_size-1)/2), int(img.shape[1]+(filter_size-1)/2)
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            output[i-y_start, j-x_start] = np.std(img_padded[i-y_start:i+y_start+1, j-y_start:j+y_start+1])
    return output