import os
import sys
import glob
import argparse
import random
from wearglasses import FaceGlasser, GLASS_PATH_LIST
from wearmask import FaceMasker, MASK_PATH_LIST

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def wear_something(pic_path ,save_pic_path,wear_type):
    if not os.path.exists(pic_path):
        print(f'Picture {pic_path} not exists.')
        sys.exit(1)
    
    if wear_type == 'glasses':
        mask_path = random.choice(GLASS_PATH_LIST)
        FaceGlasser(pic_path, mask_path, True, 'cnn',save_pic_path).mask()
    elif wear_type == 'mask':
        mask_path = random.choice(MASK_PATH_LIST)
        FaceMasker(pic_path, mask_path, True, 'cnn',save_pic_path).mask()


def get_args():
    parser = argparse.ArgumentParser(description='Wear something in the given picture.')
    parser.add_argument('--path', default='./imgs', help='path to put something')
    parser.add_argument('--wear_type', default='glasses', choices=['mask', 'glasses'], help='Which type to wear')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    imgs_dir = args.path
    files = glob.glob(os.path.join(imgs_dir, '*'))

    for i, file in enumerate(files):
        print("{}/{} {}".format(i, len(files), file))
        save_path = os.path.join('results', os.path.basename(file))
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        wear_something(file,save_path,wear_type=args.wear_type)