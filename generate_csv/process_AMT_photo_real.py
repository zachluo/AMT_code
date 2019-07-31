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

    fake_results = []
    rand_recon_results = []

    for r in results:
        flag = False
        for i in range(1, 51):
            if r['Input.images_left'+str(i)].split('/')[0].split('_')[-1] == '1':
                flag = True
            if r['Input.images_right'+str(i)].split('/')[0].split('_')[-1] == '1':
                flag = True
        if flag:
            fake_results.append(r)
        else:
            rand_recon_results.append(r)

    print('number of fake: {}, number of rand_recon: {}'.format(len(fake_results), len(rand_recon_results)))

    fake_performance = []
    rand_recon_performance = []

    for r in fake_results:
        # if r['Answer.comments'] == 'i should get a bonus for getting 85 percent correct':
        performance = 0
        for i in range(11, 51):
            if r['Input.gt_side'+str(i)] == r['Answer.selection'+str(i)]:
                performance += 1
        fake_performance.append(performance / 40)

    for r in rand_recon_results:
        performance = 0
        for i in range(11, 51):
            if r['Input.gt_side'+str(i)] == r['Answer.selection'+str(i)]:
                performance += 1
        rand_recon_performance.append(performance / 40)

    print('fake mean: {:.2f}%, fake std: {:.2f}%, rand_recon mean: {:.2f}%, rand_recon std:{:.2f}%'.format(100*np.mean(fake_performance), 100*np.std(fake_performance), 100*np.mean(rand_recon_performance), 100*np.std(rand_recon_performance)))


if __name__ == '__main__':
    path = '/p300/FaceChange/user_study/AMT_result/real1.csv'
    process(path)