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
        img = error_diffusion(img*255)
        cv2.imwrite('halftone_secret.png', img*255)
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
