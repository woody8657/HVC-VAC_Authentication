import os
import cv2
import time
import glob
import argparse
from matplotlib.cbook import flatten
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as convolution

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fingerprint", type=str, help="fingerprint image path", required=True)
    parser.add_argument("-s", "--signature", type=str, help="fingerprint image path", required=True)
    parser.add_argument("-m", "--message", type=str, help="message image directory", required=True)
    parser.add_argument("-o", "--output", type=str, help="output images directory", required=True)
    parser.add_argument("-sh", "--shape", type=int, nargs=2, help="output image shape", default=None)
    parser.add_argument("-r", "--resample_range", type=float, nargs=2, help="range for resampling shared images", default=[0.35, 0.65])
    parser.add_argument("-v", "--verbose", type=int, help="verbosity", default=0)
    args_ = parser.parse_args()
    return args_

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
        img[img > 0.5] = 1.0
        img[img <= 0.5] = 0.0
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

class HVC_VAC():
    def __init__(self, secret, shape=None, kernel=None, max_iter=100000, verbose=1):
        timer = clock()
        timer.tick()
        self.secret = secret
        self.w_mask = self.secret == 1
        self.b_mask = self.secret == 0
        self.w_size = np.sum(self.w_mask)
        self.b_size = np.sum(self.b_mask)
        self.shape = self.secret.shape
        self.size = np.prod(self.shape)
        self.kernel = self._get_kernel(kernel)
        self.kernel_size = self.kernel.shape
        self.max_iter = max_iter
        self.verbose = verbose
        timer.tock()
        if self.verbose: print("INFO: HVC-VAC initialization done, time: %.5fs" % timer.get_time())

    def _flip01(self, arr):
        return np.logical_not(arr)

    def _gaussian_kernel(self, kernel_size=9, sigma=1.5, mean=0.0):
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return kernel / np.sum(kernel)

    def _get_kernel(self, kernel):
        if kernel is not None: return kernel
        return self._gaussian_kernel(9, sigma=1.5, mean=0.0)

    def _get_score(self, pattern, target):
        if target == 'void': pattern = self._flip01(pattern)
        score = convolution(pattern, self.kernel, mode='same', boundary='wrap')
        score[pattern == 0.0] = 0.0
        return score

    def _find_void_cluster(self, target, index, region=None):
        if target == 'cluster': 
            scores = self.cluster_scores[index]
        elif target == 'void':
            scores = self.void_scores[index]
            
        if region == 'W':
            scores[self.b_mask] = 0.0
        elif region == 'B':
            scores[self.w_mask] = 0.0
            
        max_index = np.argmax(scores)
                
        return np.unravel_index(max_index, self.shape)
    
    def run_step_0(self):
        timer = clock()
        timer.tick()
        flat_mask = np.arange(np.prod(self.secret.shape))
        
        # Select white region
        W_1 = np.random.choice(flat_mask, size=self.w_size//2, replace=False, p=self.w_mask.flatten()/np.sum(self.w_mask))
        # W_0_p = self.w_mask.flatten()
        # W_0_p[W_1] = 0.0
        # W_0 = np.random.choice(flat_mask, size=self.w_size//2, replace=False, p=W_0_p / np.sum(W_0_p))
        W_0 = np.array(list(set(np.where(self.w_mask.flatten()==True)[0])-set(W_1)))
        

        # Select black region
        B_1 = np.random.choice(flat_mask, size=self.b_size//2, replace=False, p=self.b_mask.flatten()/np.sum(self.b_mask))
        # B_0_p = self.b_mask.flatten()
        # B_0_p[B_1] = 0.0
        # B_0 = np.random.choice(flat_mask, size=self.b_size//2, replace=False, p=B_0_p / np.sum(B_0_p))
        B_0 = np.array(list(set(np.where(self.b_mask.flatten()==True)[0])-set(B_1)))
        
        # Get Rp1 & RP2
        RP1 = np.zeros(self.secret.flatten().shape)
        RP2 = np.zeros(self.secret.flatten().shape)
        # Give same      pixel value on RP1 & RP2 in white region so we would have pixel value 1 after XNOR
        # Give different pixel value on RP1 & RP2 in white region so we would have pixel value 0 after XNOR
        RP1[W_0] = 0
        RP2[W_0] = 0
        RP1[W_1] = 1
        RP2[W_1] = 1
        RP1[B_0] = 0
        RP2[B_0] = 1
        RP1[B_1] = 1
        RP2[B_1] = 0
        self.RPs = [np.reshape(RP1, self.secret.shape), np.reshape(RP2, self.secret.shape)]
        cv2.imwrite('d.png', decode_shares(self.RPs[0],self.RPs[1])*255)
        cluster_score_1 = self._get_score(self.RPs[0], 'cluster')
        cluster_score_2 = self._get_score(self.RPs[1], 'cluster')
        self.cluster_scores = [cluster_score_1, cluster_score_2]
        void_score_1 = self._get_score(self.RPs[0], 'void')
        void_score_2 = self._get_score(self.RPs[1], 'void')
        self.void_scores = [void_score_1, void_score_2]
        
        timer.tock()
        if self.verbose: print("INFO: Step 0 done, time: %.5fs" % timer.get_time())      

    def _update_void_cluster_score(self, prototype, pos, index, target=None):
        x, y = pos
        kx_, ky_ = self.kernel_size
        kx = kx_ // 2
        ky = ky_ // 2
        pad_prototype = np.pad(prototype, (2*kx, 2*ky), mode='wrap')
        px_min, px_max = x, x + 4*kx + 1
        py_min, py_max = y, y + 4*ky + 1
        patch = pad_prototype[px_min:px_max, py_min:py_max]
        offset_x, offset_y = (x - kx) % self.shape[0], (y - ky) % self.shape[1]
        
        # Cluster score update
        if target is None or target == 'cluster':
            cluster_patch = patch
            cluster_patch_score = convolution(cluster_patch, self.kernel, mode='valid')
            cluster_patch_score[cluster_patch[kx:-kx, ky:-ky] == 0.0] = 0.0
            # First roll the score matrix to move the update patch to (0, 0)
            # purpose for this is to avoid overflow wrap around problems
            cluster_shift_score = np.roll(np.roll(self.cluster_scores[index], -offset_x, axis=0), -offset_y, axis=1)
            cluster_shift_score[:kx_, :ky_] = cluster_patch_score
            self.cluster_scores[index] = np.roll(np.roll(cluster_shift_score, offset_x, axis=0), offset_y, axis=1)
        
        # Void score update
        if target is None or target == 'void':
            void_patch = self._flip01(patch)
            void_patch_score = convolution(void_patch, self.kernel, mode='valid')
            void_patch_score[void_patch[kx:-kx, ky:-ky] == 0.0] = 0.0
            # First roll the score matrix to move the update patch to (0, 0)
            # purpose for this is to avoid overflow wrap around problems
            void_shift_score = np.roll(np.roll(self.void_scores[index], -offset_x, axis=0), -offset_y, axis=1)
            void_shift_score[:kx_, :ky_] = void_patch_score
            self.void_scores[index] = np.roll(np.roll(void_shift_score, offset_x, axis=0), offset_y, axis=1)
            
    def run_step_1(self):
        timer = clock()
        SPs = [self.RPs[0].copy(), self.RPs[1].copy()]
        timer.tick()
        
        for i in range(self.max_iter):
            # Select RP to run on
            index = np.random.randint(2)
            region = None
            
            # Find tightest cluster on selected RP
            cluster_pos = self._find_void_cluster('cluster', index=index, region=region)
            SPs[index][cluster_pos] = 0.0
            self._update_void_cluster_score(SPs[index], cluster_pos, index=index)
            if self.secret[cluster_pos] == 1.0: region = 'W'
            else: region = 'B'
            
            # Find largest void on selected RP
            void_pos = self._find_void_cluster('void', index=index, region=region)
            SPs[index][void_pos] = 1.0
            self._update_void_cluster_score(SPs[index], void_pos, index=index)
            
            # Repeat on the other RP
            index = int(not index)
            if region == 'W':
                SPs[index][cluster_pos] = 0.0
                self._update_void_cluster_score(SPs[index], cluster_pos, index=index)
                SPs[index][void_pos] = 1.0
                self._update_void_cluster_score(SPs[index], void_pos, index=index)
            else:
                SPs[index][cluster_pos] = 1.0
                self._update_void_cluster_score(SPs[index], cluster_pos, index=index)
                SPs[index][void_pos] = 0.0
                self._update_void_cluster_score(SPs[index], void_pos, index=index)
            
            # Check termination
            if cluster_pos == void_pos: break
        self.SPs = SPs
                
        timer.tock()
        if self.verbose: print("INFO: Step 1 done, time: %.5fs" % timer.get_time())
    
    def run_step_2(self):
        timer = clock()
        timer.tick()
        dither_matrix_1 = self.vac_operation_2(0)
        dither_matrix_2 = self.vac_operation_2(1)
        self.TAs = [dither_matrix_1, dither_matrix_2]
        timer.tock()
        if self.verbose: print("INFO: Step 2 done, time: %.5fs" % timer.get_time())

    def vac_operation_2(self, index):
        # dither matrix generation
        dither_matrix = np.zeros(self.shape, dtype=float)
        ones = int(np.sum(self.SPs[index]))
        
        # Phase I
        pattern = self.SPs[index].copy()
        self.cluster_score = self._get_score(pattern, target='cluster')
        for rank in reversed(range(ones)):
            cluster_pos = self._find_void_cluster(target='cluster', index=index)
            pattern[cluster_pos] = 0
            self._update_void_cluster_score(pattern, cluster_pos, index=index, target='cluster')
            dither_matrix[cluster_pos] = rank
            
        # Phase II
        pattern = self.SPs[index].copy()
        self.void_score = self._get_score(pattern, 'void')
        for rank in range(ones, self.size):
            void_pos = self._find_void_cluster(target='void', index=index)
            pattern[void_pos] = 1
            self._update_void_cluster_score(pattern, void_pos, index=index, target='void')
            dither_matrix[void_pos] = rank
            
        # Normalize
        dither_matrix_norm = dither_matrix / self.size
        
        return dither_matrix_norm

    def run(self):
        self.run_step_0()
        self.run_step_1()
        self.run_step_2()
        
    def halftone(self, image, index, resample_range=(0.125, 0.875)):
        image = cv2.resize(image, (self.shape[1], self.shape[0]))
        image = image * np.abs(np.subtract(resample_range[1], resample_range[0])) + resample_range[0]
        halftone_img = np.zeros(image.shape)
        halftone_img[image > self.TAs[index]] = 1.0
        return halftone_img

if __name__ == "__main__":
    args = get_args()

    # Load images
    fingerprint = read_image(args.fingerprint, resize=args.shape, binary=True)
    signature = read_image(args.signature, resize=args.shape)
    messages = glob.glob(os.path.join(args.message, '*'))
    messages = [read_image(p, resize=args.shape) for p in messages]
    messages_num = len(messages)

    # Run HVC-VAC
    hvc_vac = HVC_VAC(fingerprint, shape=args.shape, verbose=args.verbose)
    hvc_vac.run()

    # Get halftoned images
    ht_signature = hvc_vac.halftone(signature, 0, resample_range=args.resample_range)
    ht_messages = [hvc_vac.halftone(m, 1, resample_range=args.resample_range) for m in messages]
    ht_fingerprints = [decode_shares(ht_signature, m) for m in ht_messages]

    # Save images
    save_image(os.path.join(args.output, 'signature.png'), ht_signature)
    [save_image(os.path.join(args.output, 'message_%d.png' % i), ht_messages[i]) for i in range(messages_num)]
    [save_image(os.path.join(args.output, 'fingerprint_%d.png' % i), ht_fingerprints[i]) for i in range(messages_num)]

    # Check if signatures are identical
    signatures = [decode_shares(ht_messages[i], ht_fingerprints[i]) for i in range(messages_num)]
    for i in range(messages_num-1):
        for j in range(i+1, messages_num):
            similarity = np.sum(signatures[i] == signatures[j]) / np.prod(args.shape) * 100
            print("Same pixel value ratio of signature %d & %d: %.2f %%" % (i, j, similarity))

# python HVC-VAC.py -f fingerprint.png -s signature.png -m messages -o outputs -sh 200 200 -r 0.35 0.65 -v 1