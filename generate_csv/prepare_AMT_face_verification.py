import glob
import numpy as np
import os
import csv
import pickle
import random
import shutil

random.seed(2019)
number_samples = 30

def sample_with_same_id(img_id, img_name, casia_path):
    samples = glob.glob(os.path.join(casia_path, img_id, '*.jpg'))
    while True:
        idx = random.randint(0, len(samples) - 1)
        sample_name = samples[idx].split('/')[-1].split('.')[0]
        if sample_name != img_name:
            break
    return samples[idx]

def process(img_paths, url):
    with open(img_paths, 'rb') as f:
        img_paths = pickle.load(f)[:number_samples]

    assignments = []
    for idx, img_path in enumerate(img_paths):
        print(img_path)
        img_id, img_name = img_path.split('/')[-2:]
        img_name = img_name.split('.')[0]

        # real vs fake
        # other_real = sample_with_same_id(img_id, img_name, casia_path)
        # other_real_img_name = '_'.join([img_id, img_name, 'real_fake']) + '.jpg'
        # shutil.copy(other_real, os.path.join(result_path, other_real_img_name))
        # assignments.append([os.path.join(url, other_real_img_name), os.path.join(url, '_'.join([img_id, img_name, 'fake']) + '.png'), 'real_fake'])
        assignments.append([os.path.join(url, '_'.join([img_id, img_name, 'real']) + '.png'),
                            os.path.join(url, '_'.join([img_id, img_name, 'fake']) + '.png'), 'real_fake', idx*4])

        # real vs recon
        # other_real = sample_with_same_id(img_id, img_name, casia_path)
        # other_real_img_name = '_'.join([img_id, img_name, 'real_recon']) + '.jpg'
        # shutil.copy(other_real, os.path.join(result_path, other_real_img_name))
        # assignments.append([os.path.join(url, other_real_img_name), os.path.join(url, '_'.join([img_id, img_name, 'recon']) + '.png'), 'real_recon'])
        assignments.append([os.path.join(url, '_'.join([img_id, img_name, 'real']) + '.png'),
                            os.path.join(url, '_'.join([img_id, img_name, 'recon']) + '.png'), 'real_recon', idx*4+1])

        # real vs rand_recon
        # other_real = sample_with_same_id(img_id, img_name, casia_path)
        # other_real_img_name = '_'.join([img_id, img_name, 'real_rand_recon']) + '.jpg'
        # shutil.copy(other_real, os.path.join(result_path, other_real_img_name))
        # assignments.append([os.path.join(url, other_real_img_name), os.path.join(url, '_'.join([img_id, img_name, 'rand_recon']) + '.png'), 'real_rand_recon'])
        assignments.append([os.path.join(url, '_'.join([img_id, img_name, 'real']) + '.png'),
                            os.path.join(url, '_'.join([img_id, img_name, 'rand_recon']) + '.png'), 'real_rand_recon', idx*4+2])

        # fake vs rand_fake
        assignments.append([os.path.join(url, '_'.join([img_id, img_name, 'fake']) + '.png'),
                            os.path.join(url, '_'.join([img_id, img_name, 'rand_fake']) + '.png'), 'fake_rand_fake', idx*4+3])

    return assignments

if __name__ == '__main__':
    # casia_path = '/p300/dataset/casia/images'
    # result_path = 'user_study/user_study_cos_5th_1Feat_1wrFeat_0reconFeat_2FR_1WR_1dis_1GAN_1recon_1M_2WR_10L1_10randreconL1_100recon_0318082130/test_web/images/epoch5'
    img_paths = 'img_paths.pickle'
    url = 'https://raw.githubusercontent.com/zachluo/AMT/master/face_verification'
    shuffle = True
    assignments = process(img_paths, url)

    if shuffle:
        random.shuffle(assignments)

    with open('batch.csv', 'w', newline='') as f:
        spamwriter = csv.writer(f, delimiter=',')
        spamwriter.writerow(['image_url', 'img_a', 'img_b', 'task', 'id'])
        for a in assignments:
            spamwriter.writerow([''] + a)
