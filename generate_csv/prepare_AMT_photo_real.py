import glob
import numpy as np
import os
import csv
import pickle
import random
import shutil
from PIL import Image

random.seed(2019)

def checkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def process(path, real_dir, fake_dir, rand_recon_dir, number_person):
    images = glob.glob(os.path.join(path, '*.png'))
    all_people = {}
    for image in images:
        id, index = image.split('/')[-1].split('_')[:2]
        prefix = '_'.join([id, index])
        if prefix not in all_people.keys():
            all_people[prefix] = 1
        else:
            all_people[prefix] += 1
    assert len(all_people.keys()) == number_person
    for i, k in enumerate(all_people.keys()):
        assert all_people[k] == 6
        src = os.path.join(path, '_'.join([k, 'real.png']))
        dst = os.path.join(real_dir, str(i + 1) + '.jpg')
        Image.open(src).save(dst)

        src = os.path.join(path, '_'.join([k, 'fake.png']))
        dst = os.path.join(fake_dir, str(i + 1) + '.jpg')
        Image.open(src).save(dst)

        src = os.path.join(path, '_'.join([k, 'rand_recon.png']))
        dst = os.path.join(rand_recon_dir, str(i + 1) + '.jpg')
        Image.open(src).save(dst)

if __name__ == '__main__':
    number_person = 50
    path = '/p300/FaceChange/user_study/user_study_cos_5th_1Feat_1wrFeat_0reconFeat_1FR_1WR_1dis_1GAN_1recon_1M_1WR_10L1_10randreconL1_100recon_resize_conv_0321085212/test_web/images/select_50_person1'
    real_dir = '/p300/FaceChange/user_study/user_study_cos_5th_1Feat_1wrFeat_0reconFeat_1FR_1WR_1dis_1GAN_1recon_1M_1WR_10L1_10randreconL1_100recon_resize_conv_0321085212/test_web/images/select_50_person1_real'
    fake_dir = '/p300/FaceChange/user_study/user_study_cos_5th_1Feat_1wrFeat_0reconFeat_1FR_1WR_1dis_1GAN_1recon_1M_1WR_10L1_10randreconL1_100recon_resize_conv_0321085212/test_web/images/select_50_person1_fake'
    rand_recon_dir = '/p300/FaceChange/user_study/user_study_cos_5th_1Feat_1wrFeat_0reconFeat_1FR_1WR_1dis_1GAN_1recon_1M_1WR_10L1_10randreconL1_100recon_resize_conv_0321085212/test_web/images/select_50_person1_rand_recon'

    process(path, checkdir(real_dir), checkdir(fake_dir), checkdir(rand_recon_dir), number_person)


