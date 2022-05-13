import os
import argparse
import numpy as np
from HVC_VAC import HVC_VAC
from utils import read_image, decode_shares, save_image

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


def whole_process(args):
    # Load images
    fingerprint = read_image(args.fingerprint, resize=args.shape, binary=True)
    signature = read_image(args.signature, resize=args.shape)
    messages = [args.message]
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
    save_image(os.path.join(args.output, 'share1.png'), ht_signature)
    [save_image(os.path.join(args.output, 'share2%d.png' % i), ht_messages[i]) for i in range(messages_num)]
    [save_image(os.path.join(args.output, 'decode_%d.png' % i), ht_fingerprints[i]) for i in range(messages_num)]

    # Check if signatures are identical
    signatures = [decode_shares(ht_messages[i], ht_fingerprints[i]) for i in range(messages_num)]
    for i in range(messages_num-1):
        for j in range(i+1, messages_num):
            similarity = np.sum(signatures[i] == signatures[j]) / np.prod(args.shape) * 100
            print("Same pixel value ratio of signature %d & %d: %.2f %%" % (i, j, similarity))


if __name__ == "__main__":
    args = get_args()

    whole_process(args)
 
    


