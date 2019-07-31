import csv
import numpy as np

def process(path):
    results = []
    with open(path) as f:
        spamreader = csv.DictReader(f)
        for row in spamreader:
            results.append(row)

    work_id = []
    for r in results:
        work_id.append(r['WorkerId'])
    print('there are totally {} unique workers'.format(len(set(work_id))))

    real_fake_results = []
    real_recon_results = []
    real_rand_recon_results = []
    fake_rand_fake_results = []

    for r in results:
        if r['Input.task'] == 'real_fake' and r['Answer.category.label'] != 'Not sure':
            real_fake_results.append(r['Answer.category.label'])
        elif r['Input.task'] == 'real_recon' and r['Answer.category.label'] != 'Not sure':
            real_recon_results.append(r['Answer.category.label'])
        elif r['Input.task'] == 'real_rand_recon' and r['Answer.category.label'] != 'Not sure':
            real_rand_recon_results.append(r['Answer.category.label'])
        elif r['Input.task'] == 'fake_rand_fake' and r['Answer.category.label'] != 'Not sure':
            fake_rand_fake_results.append(r['Answer.category.label'])
        else:
            continue

    correct = 0
    for r in real_fake_results:
        if r == 'Yes':
            correct += 1
        elif r == 'No':
            continue
        else:
            raise NotImplementedError
    print('real_fake accuracy: {}'.format(correct / len(real_fake_results)))

    correct = 0
    for r in real_recon_results:
        if r == 'Yes':
            correct += 1
        elif r == 'No':
            continue
        else:
            raise NotImplementedError
    print('real_recon accuracy: {}'.format(correct / len(real_recon_results)))

    correct = 0
    for r in real_rand_recon_results:
        if r == 'Yes':
            correct += 1
        elif r == 'No':
            continue
        else:
            raise NotImplementedError
    print('real_rand_recon accuracy: {}'.format(correct / len(real_rand_recon_results)))

    correct = 0
    for r in fake_rand_fake_results:
        if r == 'Yes':
            correct += 1
        elif r == 'No':
            continue
        else:
            raise NotImplementedError
    print('fake_rand_fake accuracy: {}'.format(correct / len(fake_rand_fake_results)))


if __name__ == '__main__':
    path = '/p300/FaceChange/user_study/AMT_result/Batch_3578197_batch_results.csv'
    process(path)