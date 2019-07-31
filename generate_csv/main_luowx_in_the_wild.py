# system libraries
import os, sys
import os.path as osp
import time
import shutil
import numpy as np
from PIL import Image
import gc
from collections import OrderedDict
import GPUtil
import tqdm

import torch
import torchvision.transforms as transforms

# libraries within this package
from cmd_args import parse_args
from utils.tools import *
from utils.image_pool import ImagePool
from utils.visualizer_luowx import Visualizer
from utils.util import print_param_info, save_model, set_requires_grad
import datasets
import models

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def alignment(src_pts):
    # TODO: different processing in Jason's code, here, and the paper (paper concatenate multi-scale cropping)

    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    # crop_size = (96, 112)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    s = s / 125. - 1.
    r[:, 0] = r[:, 0] / 48. - 1
    r[:, 1] = r[:, 1] / 56. - 1

    all_tfms = np.empty((s.shape[0], 2, 3), dtype=np.float32)
    for idx in range(s.shape[0]):
        all_tfms[idx, :, :] = models.get_similarity_transform_for_cv2(r, s[idx, ...])
    all_tfms = torch.from_numpy(all_tfms).to(torch.device('cuda:0'))
    return all_tfms


def main():
    # # check/wait GPU is free
    # allocated_ids = [int(item) for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    # gpu_free = False
    # while not gpu_free:
    #     tmp_gpu_free = True
    #     for gpu_id in allocated_ids:
    #         # print(gpu_id, len(GPUtil.getGPUs()))
    #         mem_used = GPUtil.getGPUs()[gpu_id].memoryUsed
    #         if mem_used > 1000:
    #             # print('mem used', gpu_id, mem_used)
    #             tmp_gpu_free = False
    #             break
    #     gpu_free = tmp_gpu_free
    #     if not gpu_free:
    #         time.sleep(300)

    # parse args
    global args
    args = parse_args(sys.argv[1])
    args.test_size = 512

    # -------------------- default arg settings for this model --------------------
    # TODO: find better way for defining model-specific default args
    if hasattr(args, 'norm'):
        args.normD = args.norm
        args.normQ = args.norm
        args.normG = args.norm


    if hasattr(args, 'lambda_D_GAN') and args.lambda_D_GAN != 1.:
        """ process deprecated lambda_D_GAN """
        args.lambda_GAN = args.lambda_D_GAN
        assert args.lambda_D_GAN == args.lambda_G_GAN

    # add timestamp to ckpt_dir
    # if not args.debug:
    args.timestamp = time.strftime('%m%d%H%M%S', time.localtime())
    args.ckpt_dir += '_' + args.timestamp
    if args.lambda_G_recon > 0:
        args.display_ncols = 5 if args.lambda_dis > 0 else 3
        if args.lambda_dis > 0 and args.lambda_G_rand_recon > 0:
            args.display_ncols += 1
    else:
        args.display_ncols = 3 if args.lambda_dis > 0 else 2



    # !!! FINISH defining args before logging args
    # -------------------- init ckpt_dir, logging --------------------
    os.makedirs(args.ckpt_dir, mode=0o777, exist_ok=True)

    # -------------------- init visu --------------------
    visualizer = Visualizer(args)

    # logger = Logger(osp.join(args. ckpt_dir, 'log'))
    visualizer.logger.log('sys.argv:\n' + ' '.join(sys.argv))
    for arg in sorted(vars(args)):
        visualizer.logger.log('{:20s} {}'.format(arg, getattr(args, arg)))
    visualizer.logger.log('')

    # -------------------- code copy --------------------
    # TODO: find better approach
    # copy config yaml
    shutil.copyfile(sys.argv[1], osp.join(args.ckpt_dir, osp.basename(sys.argv[1])))

    repo_basename = osp.basename(osp.dirname(osp.abspath(__file__)))
    repo_path = osp.join(args.ckpt_dir, repo_basename)
    os.makedirs(repo_path, mode=0o777, exist_ok=True)

    # walk_res = os.walk('.')
    # useful_paths = [path for path in walk_res if
    #                 '.git' not in path[0] and
    #                 'checkpoints' not in path[0] and
    #                 'configs' not in path[0] and
    #                 '__pycache__' not in path[0] and
    #                 'tee_dir' not in path[0] and
    #                 'tmp' not in path[0]]
    # # print('useful_paths', useful_paths)
    # for p in useful_paths:
    #     for item in p[-1]:
    #         if not (item.endswith('.py') or item.endswith('.c') or item.endswith('.h') or item.endswith('.md')):
    #             continue
    #         old_path = osp.join(p[0], item)
    #         new_path = osp.join(repo_path, p[0][2:], item)
    #         basedir = osp.dirname(new_path)
    #         os.makedirs(basedir, mode=0o777, exist_ok=True)
    #         shutil.copyfile(old_path, new_path)

    # if args.evaluate:
    #     shutil.copyfile(args.resume, osp.join(args.ckpt_dir, 'model_used.pth.tar'))
    # If cannot find file, will raise FileNotFoundError
    # The destination location must be writable; otherwise, an OSError exception will be raised.
    #  If dst already exists, it will be replaced. Special files such as character or block devices
    #  and pipes cannot be copied with this function.

    # -------------------- dataset & loader --------------------
    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        transform=transforms.Compose([
            transforms.Resize(args.imageSize, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ]),
        args=args
    )
    visualizer.logger.log('train_dataset: ' + str(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    if not args.debug:
        args.html_iter_freq = len(train_loader) // args.html_per_train_epoch
        visualizer.logger.log('change args.html_iter_freq to %s' % args.html_iter_freq)
        args.save_iter_freq = len(train_loader) // args.html_per_train_epoch
        visualizer.logger.log('change args.save_iter_freq to %s' % args.html_iter_freq)

    test_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.Compose([
            transforms.Resize(args.imageSize, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ]),
        args=args
    )

    visualizer.logger.log('test_dataset: ' + str(test_dataset))
    visualizer.logger.log('test img paths:')
    for anno in test_dataset.raw_annotations:
        visualizer.logger.log('%s %d %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f' % (anno[0], anno[1], anno[2], anno[3], anno[4], anno[5], anno[6], anno[7], anno[8], anno[9], anno[10], anno[11]))
    visualizer.logger.log('')

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)),
        drop_last=True
    )
    #assert len(test_loader) == 1
    print('test_loader has {} images'.format(len(test_loader)))

    # --------------------------------------------------------------------------------
    # -------------------- create model --------------------
    # visualizer.logger.log("=>  creating model '{}'".format(args.arch))

    args.gpu_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
    args.device = torch.device('cuda:0') if args.gpu_ids else torch.device('cpu')

    model_dict = {}
    model_dict['D_nets'] = []
    model_dict['G_nets'] = []

    # D, Q
    if args.lambda_dis > 0:
        if args.recon_pair_GAN:
            infogan_func = models.define_infoGAN_pair_D
        else:
            infogan_func = models.define_infoGAN
        model_dict['D'], model_dict['Q'] = infogan_func(
            args.output_nc,
            args.ndf,
            args.which_model_netD,
            args.n_layers_D,
            args.n_layers_Q,
            16,
            args.passwd_length // 4,
            args.normD,
            args.normQ,
            args.init_type,
            args.init_gain,
            args.gpu_ids,
            args.use_old_Q,
            args.use_minus_Q)

        model_dict['G_nets'].append(model_dict['Q'])
        if args.lambda_GAN == 0:
            del model_dict['D']
        else:
            model_dict['D_nets'].append(model_dict['D'])
    else:
        if args.lambda_GAN > 0:
            model_dict['D'] = models.define_D(args.input_nc, args.ndf, args.which_model_netD, args.n_layers_D,
                                   args.normD, args.no_lsgan,
                                   args.init_type, args.init_gan,
                                   args.gpu_ids)
            model_dict['D_nets'].append(model_dict['D'])

    # G
    if 'with_noise' in args.which_model_netG or args.lambda_dis == 0.:
        G_input_nc = args.input_nc
    else:
        G_input_nc = args.input_nc + args.passwd_length

    model_dict['G'] = models.define_G(G_input_nc, args.output_nc,
                                      args.ngf, args.which_model_netG, args.n_downsample_G,
                                      args.normG, not args.no_dropout,
                                      args.init_type, args.init_gain,
                                      args.gpu_ids,
                                      args.passwd_length,
                                      use_leaky=args.use_leakyG,
                                      use_resize_conv=args.use_resize_conv)
    model_dict['G_nets'].append(model_dict['G'])

    # D_pair
    if args.lambda_pair_GAN > 0:
        model_dict['pair_D'] = models.define_D(args.input_nc * 2, args.ndf, args.which_model_netD, args.n_layers_D,
                                               args.normD, args.no_lsgan,
                                               args.init_type, args.init_gain,
                                               args.gpu_ids)
        model_dict['D_nets'].append(model_dict['pair_D'])

    # FR
    netFR = models.sphere20a(feature=args.feature_layer)
    if len(args.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netFR.to(args.gpu_ids[0])
        netFR = torch.nn.DataParallel(netFR, args.gpu_ids)
    netFR.module.load_state_dict(torch.load('./pretrained_models/sphere20a_20171020.pth', map_location='cpu'))
    model_dict['FR'] = netFR
    model_dict['D_nets'].append(netFR)

    visualizer.logger.log('model_dict')
    for k, v in model_dict.items():
        visualizer.logger.log(k+':')
        if isinstance(v, list):
            visualizer.logger.log('list, len: ' + str(len(v)))
            for item in v:
                visualizer.logger.log(item.module.__class__.__name__, end=' ')
            visualizer.logger.log('')
        else:
            visualizer.logger.log(v)

    # -------------------- criterions --------------------
    criterion_dict = {
        'GAN': models.GANLoss(args.gan_mode).to(args.device),
        'FR': models.AngleLoss().to(args.device),
        'L1': torch.nn.L1Loss().to(args.device),
        'DIS': torch.nn.CrossEntropyLoss().to(args.device),
        'Feat': torch.nn.CosineEmbeddingLoss().to(args.device) if args.feature_loss == 'cos' else torch.nn.MSELoss().to(args.device)
    }
    # -------------------- optimizers --------------------
    # considering separate optimizer for each network?
    optimizer_G_params = [{'params': model_dict['G'].parameters(), 'lr': args.lr}]
    if args.lambda_dis > 0:
        optimizer_G_params.append({'params': model_dict['Q'].parameters(), 'lr': args.lr})

    optimizer_G = torch.optim.Adam(optimizer_G_params,
                                   lr=args.lr,
                                   betas=(args.beta1, 0.999),
                                   weight_decay=args.weight_decay)

    optimizer_D_params = []
    if args.lambda_GAN > 0:
        optimizer_D_params.append({'params': model_dict['D'].parameters(), 'lr': args.lr})
    if not args.fix_FR and args.lambda_FR > 0:
        optimizer_D_params.append({'params': netFR.parameters(), 'lr': args.lr * 0.1})
    if args.lambda_pair_GAN > 0:
        optimizer_D_params.append({'params':model_dict['pair_D'].parameters(), 'lr': args.lr})

    if len(optimizer_D_params):
        optimizer_D = torch.optim.Adam(optimizer_D_params,
                                       betas=(args.beta1, 0.999),
                                       weight_decay=args.weight_decay)
    else:
        optimizer_D = None

    optimizer_dict = {
        'G': optimizer_G,
        'D': optimizer_D
    }

    fake_pool = ImagePool(args.pool_size)
    recon_pool = ImagePool(args.pool_size)
    fake_pair_pool = ImagePool(args.pool_size)
    WR_pair_pool = ImagePool(args.pool_size)

    if args.resume:
        if osp.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1

            for name, net in model_dict.items():
                if isinstance(net, list):
                    continue
                if hasattr(args, 'not_resume_models') and (name in args.not_resume_models):
                    continue
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                if 'state_dict_' + name in checkpoint:
                    try:
                        net.load_state_dict(checkpoint['state_dict_' + name])
                    except Exception as e:
                        visualizer.logger.log('fail to load model '+name+' '+str(e))
                else:
                    visualizer.logger.log('model '+name+' not in checkpoints, just skip')

            if args.resume_optimizer:
                for name, optimizer in optimizer_dict.items():
                    if 'optimizer_' + name in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_' + name])
                    else:
                        visualizer.logger.log('optimizer ' + name + ' not in checkpoints, just skip')

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        gc.collect()
        # torch.cuda.empty_cache()

    torch.backends.cudnn.benchmark = True

    # -------------------- miscellaneous --------------------

    if args.lambda_dis > 0:
        fixed_z, fixed_dis_target, fixed_rand_z, fixed_rand_dis_target = generate_code(args.passwd_length, args.batch_size, args.device, inv=False)
        print(fixed_z)
    else:
        fixed_z, fixed_rand_z = None, None

    # for epoch in range(args.start_epoch, args.num_epochs):
    #     print('epoch', epoch)
    #     # train
    #     if args.lambda_dis > 0:
    #         model_dict['Q'].train()
    #     if args.lambda_GAN > 0:
    #         model_dict['D'].train()
    #     if args.lambda_pair_GAN > 0:
    #         model_dict['pair_D'].train()
    #     model_dict['G'].train()
    #     if not args.fix_FR:
    #         model_dict['FR'].train()
    #
    #     epoch_start_time = time.time()
    #     train(train_loader, model_dict, criterion_dict, optimizer_dict, fake_pool, recon_pool, fake_pair_pool, WR_pair_pool, visualizer, epoch, args, test_loader, fixed_z, fixed_rand_z)
    #     epoch_time = time.time() - epoch_start_time
    #     message = 'epoch %s total time %s\n' % (epoch, epoch_time)
    #     visualizer.logger.log(message)
    #
    #     gc.collect()
    #     # torch.cuda.empty_cache()
    #
    #     # save model
    #     if epoch % args.save_epoch_freq == 0:
    #         save_model(epoch, model_dict, optimizer_dict, args, iter=len(train_loader))
    #     # test visualization
    #     if epoch % args.html_epoch_freq == 0:
    test(test_loader, model_dict, criterion_dict, visualizer, 5, args, fixed_z, fixed_rand_z, 3069)
            # gc.collect()
            # torch.cuda.empty_cache()


def unsigned_long_to_binary_repr(unsigned_long, passwd_length):
    batch_size = unsigned_long.shape[0]
    target_size = passwd_length // 4

    binary = np.empty((batch_size, passwd_length), dtype=np.float32)
    for idx in range(batch_size):
        binary[idx, :] = np.array([int(item) for item in bin(unsigned_long[idx])[2:].zfill(passwd_length)])

    dis_target = np.empty((batch_size, target_size), dtype=np.long)
    for idx in range(batch_size):
        tmp = unsigned_long[idx]
        for byte_idx in range(target_size):
            dis_target[idx, target_size - 1 - byte_idx] = tmp % 16
            tmp //= 16
    return binary, dis_target


def generate_code(passwd_length, batch_size, device, inv):
    unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
    binary, dis_target = unsigned_long_to_binary_repr(unsigned_long, args.passwd_length)
    z = torch.from_numpy(binary).to(device)
    dis_target = torch.from_numpy(dis_target).to(device)

    repeated = True
    while repeated:
        rand_unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
        repeated = np.any(unsigned_long - rand_unsigned_long == 0)
    rand_binary, rand_dis_target = unsigned_long_to_binary_repr(rand_unsigned_long, args.passwd_length)
    rand_z = torch.from_numpy(rand_binary).to(device)
    rand_dis_target = torch.from_numpy(rand_dis_target).to(device)

    if not inv:
        if args.use_minus_one:
            z = (z - 0.5) * 2
            rand_z = (rand_z - 0.5) * 2

        return z, dis_target, rand_z, rand_dis_target
    else:
        inv_unsigned_long = 2 ** args.passwd_length - 1 - unsigned_long
        inv_binary, inv_dis_target = unsigned_long_to_binary_repr(inv_unsigned_long, args.passwd_length)

        inv_z = torch.from_numpy(inv_binary).to(device)
        inv_dis_target = torch.from_numpy(inv_dis_target).to(device)

        repeated = True
        while repeated:
            another_rand_unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
            repeated = np.any(inv_unsigned_long - another_rand_unsigned_long == 0)
        another_rand_binary, another_rand_dis_target = unsigned_long_to_binary_repr(another_rand_unsigned_long, args.passwd_length)
        another_rand_z = torch.from_numpy(another_rand_binary).to(device)
        another_rand_dis_target = torch.from_numpy(another_rand_dis_target).to(device)

        if args.use_minus_one:
            z = (z - 0.5) * 2
            rand_z = (rand_z - 0.5) * 2
            inv_z = z * -1.
            another_rand_z = (another_rand_z - 0.5) * 2

        return z, dis_target, rand_z, rand_dis_target,\
               inv_z, inv_dis_target, another_rand_z, another_rand_dis_target


def train(train_loader, model_dict, criterion_dict, optimizer_dict, fake_pool, recon_pool, fake_pair_pool, WR_pool, visualizer, epoch, args, test_loader, fixed_z, fixed_rand_z):
    iter_data_time = time.time()

    for i, (img, label, landmarks, img_path) in enumerate(train_loader):
        iter_start_time = time.time()
        if i % args.print_loss_freq == 0:
            t_data = iter_start_time - iter_data_time

        visualizer.reset()
        batch_size = img.size(0)

        if args.lambda_dis > 0:
            # -------------------- generate password --------------------
            z, dis_target, rand_z, rand_dis_target, inv_z, inv_dis_target, another_rand_z, another_rand_dis_target = generate_code(args.passwd_length, batch_size, args.device, inv=True)

            # -------------------- forward --------------------
            # TODO: whether to detach
            fake = model_dict['G'](img, z.cpu())
            rand_fake = model_dict['G'](img, rand_z.cpu())
            if args.lambda_G_recon > 0:
                recon = model_dict['G'](fake, inv_z)
                rand_recon = model_dict['G'](fake, another_rand_z)
        else:
            fake = model_dict['G'](img)
            if args.lambda_G_recon > 0:
                recon = model_dict['G'](fake)

        # FR forward and FR losses
        theta = alignment(landmarks)
        grid = torch.nn.functional.affine_grid(theta, torch.Size((batch_size, 3, 112, 96)))
        real_aligned = torch.nn.functional.grid_sample(img.cuda(), grid)
        real_aligned = real_aligned[:, [2, 1, 0], ...]

        fake_aligned = torch.nn.functional.grid_sample(fake, grid)
        fake_aligned = fake_aligned[:, [2, 1, 0], ...]

        rand_fake_aligned = torch.nn.functional.grid_sample(rand_fake, grid)
        rand_fake_aligned = rand_fake_aligned[:, [2, 1, 0, ], ...]
        # (B, 3, h, w)

        if args.lambda_G_recon > 0:
            recon_aligned = torch.nn.functional.grid_sample(recon, grid)
            recon_aligned = recon_aligned[:, [2, 1, 0], ...]
            rand_recon_aligned = torch.nn.functional.grid_sample(rand_recon, grid)
            rand_recon_aligned = rand_recon_aligned[:, [2, 1, 0], ...]

        current_losses = {}
        # -------------------- D PART --------------------
        if optimizer_dict['D'] is not None:
            set_requires_grad(model_dict['G_nets'], False)
            set_requires_grad(model_dict['D_nets'], True)
            optimizer_dict['D'].zero_grad()

            id_real = model_dict['FR'](real_aligned)[0]
            loss_D_FR_real = criterion_dict['FR'](id_real, label.to(args.device))

            cnt_FR_fake = 0.
            loss_D_FR_fake_total = 0
            if args.train_M:
                id_fake = model_dict['FR'](fake_aligned.detach())[0]
                id_rand_fake = model_dict['FR'](rand_fake_aligned.detach())[0]

                loss_D_FR_fake = criterion_dict['FR'](id_fake, label.to(args.device))
                loss_D_FR_rand_fake = criterion_dict['FR'](id_rand_fake, label.to(args.device))

                loss_D_FR_fake_total += loss_D_FR_fake + loss_D_FR_rand_fake
                cnt_FR_fake += 2.
                current_losses.update({'D_FR_fake': loss_D_FR_fake.item(),
                                       'D_FR_rand': loss_D_FR_rand_fake.item(),
                                       # 'D_FR_rand_recon': loss_D_FR_rand_recon.item()
                                       })

            if args.recon_FR:
                # TODO: rand_fake_recon FR loss?
                id_recon = model_dict['FR'](recon_aligned.detach())[0]
                loss_D_FR_recon = -criterion_dict['FR'](id_recon, label.to(args.device))
                if args.lambda_FR_WR:
                    id_rand_recon = model_dict['FR'](rand_recon_aligned.detach())[0]
                    loss_D_FR_rand_recon = criterion_dict['FR'](id_rand_recon, label.to(args.device))
                    current_losses.update({'D_FR_rand_recon': loss_D_FR_rand_recon.item()
                                           })
                else:
                    loss_D_FR_rand_recon = 0.

                loss_D_FR_fake_total += loss_D_FR_recon + args.lambda_FR_WR * loss_D_FR_rand_recon
                cnt_FR_fake += 1. + args.lambda_FR_WR
                current_losses.update({'D_FR_recon': loss_D_FR_recon.item(),
                                       # 'D_FR_rand_recon': loss_D_FR_rand_recon.item()
                                       })


            loss_D_FR_fake_avg = loss_D_FR_fake_total / float(cnt_FR_fake)

            loss_D = args.lambda_FR * (loss_D_FR_real + loss_D_FR_fake_avg) * 0.5
            current_losses.update({'D_FR_real': loss_D_FR_real.item(),
                                   'D_FR_fake': loss_D_FR_fake_avg.item()
                              # 'D_FR_fake': loss_D_FR_fake.item(),
                              # 'D_FR_rand': loss_D_FR_rand_fake.item(),
                              # 'D_FR_rand_recon': loss_D_FR_rand_recon.item()
                              })

            # GAN loss
            if args.lambda_GAN > 0:
                # real
                if args.recon_pair_GAN:
                    assert args.single_GAN_recon_only
                    real_input = torch.cat((img.cuda(), recon.detach()), dim=1)
                else:
                    real_input = img

                pred_D_real = model_dict['D'](real_input)
                loss_D_real = criterion_dict['GAN'](pred_D_real, True)

                # fake
                loss_D_fake_total = 0.
                loss_D_fake_total_weights = 0.

                # recon
                if args.lambda_GAN_recon:
                    if args.recon_pair_GAN:
                        recon_input_to_pool = torch.cat((recon.detach().cpu(), img), dim=1)
                    else:
                        recon_input_to_pool = recon.detach().cpu()

                    pred_D_recon = model_dict['D'](recon_pool.query(recon_input_to_pool))
                    loss_D_recon = criterion_dict['GAN'](pred_D_recon, False)

                    loss_D_fake_total += args.lambda_GAN_recon * loss_D_recon
                    loss_D_fake_total_weights += args.lambda_GAN_recon
                    current_losses['D_recon'] = loss_D_recon.item()

                if not args.single_GAN_recon_only:
                    assert args.lambda_pair_GAN == 0
                    if args.train_M:
                        all_M = torch.cat((fake.detach().cpu(),
                                           rand_fake.detach().cpu(),
                                           ), 0)
                        pred_D_M = model_dict['D'](fake_pool.query(all_M))
                        loss_D_M = criterion_dict['GAN'](pred_D_M, False)

                        loss_D_fake_total += args.lambda_GAN_M * loss_D_M
                        loss_D_fake_total_weights += args.lambda_GAN_M
                        current_losses['D_M'] = loss_D_M.item()

                    if args.lambda_GAN_WR:
                        pred_D_WR = model_dict['D'](WR_pool.query(rand_recon.detach().cpu()))
                        loss_D_WR = criterion_dict['GAN'](pred_D_WR, False)

                        loss_D_fake_total += args.lambda_GAN_WR * loss_D_WR
                        loss_D_fake_total_weights += args.lambda_GAN_WR
                        current_losses['D_WR'] = loss_D_WR.item()


                loss_D_fake = loss_D_fake_total / loss_D_fake_total_weights
                loss_D += args.lambda_GAN * (loss_D_fake + loss_D_real) * 0.5

                current_losses.update({
                    'D_real': loss_D_real.item(),
                    'D_fake': loss_D_fake.item()
                })


            if args.lambda_pair_GAN > 0:
                loss_pair_fake_total = 0
                loss_pair_real_total = 0
                loss_pair_cnt = 0.
                if args.train_M:
                    pred_pair_real1 = model_dict['pair_D'](torch.cat((img.cuda(), fake.detach()), 1))
                    pred_pair_real2 = model_dict['pair_D'](torch.cat((img.cuda(), rand_fake.detach()), 1))

                    all_fake_pair = torch.cat((torch.cat((fake.detach().cpu(), img), 1),
                                               torch.cat((rand_fake.detach().cpu(), img), 1),
                                               ), 0)
                    pred_pair_fake = model_dict['pair_D'](fake_pair_pool.query(all_fake_pair))

                    loss_pair_M_real = (criterion_dict['GAN'](pred_pair_real1, True) + criterion_dict['GAN'](pred_pair_real2, True)) / 2.
                    loss_pair_M_fake = criterion_dict['GAN'](pred_pair_fake, False)

                    loss_pair_real_total += loss_pair_M_real
                    loss_pair_fake_total += loss_pair_M_fake
                    loss_pair_cnt += 1

                pred_pair_WR_real = model_dict['pair_D'](torch.cat((img.cuda(), rand_recon.detach()), 1))
                pred_pair_WR_fake = model_dict['pair_D'](WR_pool.query(torch.cat((rand_recon.detach().cpu(), img), 1)))

                loss_pair_WR_real = criterion_dict['GAN'](pred_pair_WR_real, True)
                loss_pair_WR_fake = criterion_dict['GAN'](pred_pair_WR_fake, False)

                loss_pair_real_total += args.multiple_pair_WR_GAN * loss_pair_WR_real
                loss_pair_fake_total += args.multiple_pair_WR_GAN * loss_pair_WR_fake
                loss_pair_cnt += args.multiple_pair_WR_GAN

                loss_pair_D_real = loss_pair_real_total / loss_pair_cnt  # (loss_pair_M_real + args.multiple_pair_WR_GAN * loss_pair_WR_real) / (1. + args.multiple_pair_WR_GAN)
                loss_pair_D_fake = loss_pair_fake_total / loss_pair_cnt #(loss_pair_M_fake + args.multiple_pair_WR_GAN * loss_pair_WR_fake) / (1. + args.multiple_pair_WR_GAN)

                current_losses.update({
                    'pair_D_fake': loss_pair_D_fake.item(),
                    'pair_D_real': loss_pair_D_real.item()
                })
                loss_D += args.lambda_pair_GAN * (loss_pair_D_fake + loss_pair_D_real) * 0.5

            current_losses['D'] = loss_D.item()
            # D backward and optimizer steps
            loss_D.backward()

            if args.gan_mode == 'wgangp':
                real_to_wgangp = torch.cat((img, img), 0).to(args.device)
                if np.random.rand() > 0.5:
                    fake_selected = fake.detach()
                else:
                    fake_selected = rand_fake.detach()
                fake_to_wgangp = torch.cat((fake_selected, rand_recon.detach()), 0)
                loss_gp, gradients = models.cal_gradient_penalty(model_dict['D'], real_to_wgangp, fake_to_wgangp, args.device)
                # print('gradeints abs/l2 mean:', gradients[0], gradients[1])
                loss_gp *= args.lambda_GAN
                # print('loss_gp', loss_gp.item())
                loss_gp.backward()

            optimizer_dict['D'].step()

        # -------------------- G PART --------------------
        # init
        set_requires_grad(model_dict['D_nets'], False)
        set_requires_grad(model_dict['G_nets'], True)
        optimizer_dict['G'].zero_grad()

        loss_G = 0
        # GAN loss
        if args.lambda_GAN > 0:
            loss_G_GAN_total = 0.
            loss_G_GAN_total_weights = 0.

            # recon
            if args.lambda_GAN_recon:
                if args.recon_pair_GAN:
                    recon_input_G = torch.cat((recon, img.cuda()), dim=1)
                else:
                    recon_input_G = recon
                pred_G_recon = model_dict['D'](recon_input_G)
                loss_G_recon = criterion_dict['GAN'](pred_G_recon, True)

                loss_G_GAN_total += args.lambda_GAN_recon * loss_G_recon
                loss_G_GAN_total_weights += args.lambda_GAN_recon
                current_losses['G_recon'] = loss_G_recon.item()

            if not args.single_GAN_recon_only:
                if args.train_M:
                    pred_G_fake = model_dict['D'](fake)
                    pred_G_rand_fake = model_dict['D'](rand_fake)

                    loss_G_fake = criterion_dict['GAN'](pred_G_fake, True)
                    loss_G_rand_fake = criterion_dict['GAN'](pred_G_rand_fake, True)

                    loss_G_GAN_total += args.lambda_GAN_M * 0.5 * (loss_G_fake + loss_G_rand_fake)
                    loss_G_GAN_total_weights += args.lambda_GAN_M

                    current_losses['G_M'] = 0.5 * (loss_G_fake.item() + loss_G_rand_fake.item())

                pred_G_WR = model_dict['D'](rand_recon)
                loss_G_WR = criterion_dict['GAN'](pred_G_WR, True)
                current_losses['G_WR'] = loss_G_WR.item()

                loss_G_GAN_total += args.lambda_GAN_WR * loss_G_WR
                loss_G_GAN_total_weights += args.lambda_GAN_WR

            loss_G_GAN = loss_G_GAN_total / loss_G_GAN_total_weights
            loss_G += args.lambda_GAN * loss_G_GAN

            current_losses.update({'G_GAN': loss_G_GAN.item(),
                                   })


        if args.lambda_pair_GAN > 0:
            loss_pair_G_total = 0
            cnt_pair_G = 0.

            if args.train_M:
                pred_pair_fake1_G = model_dict['pair_D'](torch.cat((fake, img.cuda()), 1))
                pred_pair_fake2_G = model_dict['pair_D'](torch.cat((rand_fake, img.cuda()), 1))

                loss_pair_M_G = (criterion_dict['GAN'](pred_pair_fake1_G, True)
                               + criterion_dict['GAN'](pred_pair_fake2_G, True)) / 2.

                loss_pair_G_total += loss_pair_M_G
                cnt_pair_G += 1.

            pred_pair_fake3_G = model_dict['pair_D'](torch.cat((rand_recon, img.cuda()), 1))
            loss_pair_WR_G = criterion_dict['GAN'](pred_pair_fake3_G, True)

            loss_pair_G_total += args.multiple_pair_WR_GAN * loss_pair_WR_G
            cnt_pair_G += args.multiple_pair_WR_GAN

            loss_pair_G_avg = loss_pair_G_total / cnt_pair_G

            loss_G += args.lambda_pair_GAN * loss_pair_G_avg
            current_losses['pair_G'] = loss_pair_G_avg.item()

        # infoGAN loss
        def infoGAN_input(img1, img2):
            if args.use_minus_Q:
                return img2 - img1
            else:
                return torch.cat((img1, img2), 1)

        if args.lambda_dis > 0:
            infogan_acc = 0
            infogan_inv_acc = 0
            infogan_rand_acc = 0
            infogan_recon_rand_acc = 0

            dis_logits = model_dict['Q'](infoGAN_input(img.cuda(), fake))
            loss_G_dis = 0
            for dis_idx in range(args.passwd_length // 4):
                a = dis_logits[dis_idx].max(dim=1)[1]
                b = dis_target[:, dis_idx]
                acc = torch.eq(a, b).type(torch.float).mean()
                infogan_acc += acc.item()
                loss_G_dis += criterion_dict['DIS'](dis_logits[dis_idx], dis_target[:, dis_idx])
            infogan_acc = infogan_acc / float(args.passwd_length // 4)

            inv_dis_logits = model_dict['Q'](infoGAN_input(fake, recon))
            loss_G_inv_dis = 0
            for dis_idx in range(args.passwd_length // 4):
                a = inv_dis_logits[dis_idx].max(dim=1)[1]
                b = inv_dis_target[:, dis_idx]
                acc = torch.eq(a, b).type(torch.float).mean()
                infogan_inv_acc += acc.item()
                loss_G_inv_dis += criterion_dict['DIS'](inv_dis_logits[dis_idx], inv_dis_target[:, dis_idx])
            infogan_inv_acc = infogan_inv_acc / float(args.passwd_length // 4)

            rand_dis_logits = model_dict['Q'](infoGAN_input(img.cuda(), rand_fake))
            loss_G_rand_dis = 0
            for dis_idx in range(args.passwd_length // 4):
                a = rand_dis_logits[dis_idx].max(dim=1)[1]
                b = rand_dis_target[:, dis_idx]
                acc = torch.eq(a, b).type(torch.float).mean()
                infogan_rand_acc += acc.item()
                loss_G_rand_dis += criterion_dict['DIS'](rand_dis_logits[dis_idx], rand_dis_target[:, dis_idx])
            infogan_rand_acc = infogan_rand_acc / float(args.passwd_length // 4)

            recon_rand_dis_logits = model_dict['Q'](infoGAN_input(fake, rand_recon))
            loss_G_recon_rand_dis = 0
            for dis_idx in range(args.passwd_length // 4):
                a = recon_rand_dis_logits[dis_idx].max(dim=1)[1]
                b = another_rand_dis_target[:, dis_idx]
                acc = torch.eq(a, b).type(torch.float).mean()
                infogan_recon_rand_acc += acc.item()
                loss_G_recon_rand_dis += criterion_dict['DIS'](recon_rand_dis_logits[dis_idx], another_rand_dis_target[:, dis_idx])
            infogan_recon_rand_acc = infogan_recon_rand_acc / float(args.passwd_length // 4)

            # current_losses.update({'G_dis': loss_G_dis.item(),
            #                        'G_inv_dis': loss_G_inv_dis.item(),
            #                        'G_dis_acc': infogan_acc,
            #                        'G_inv_dis_acc': infogan_inv_acc,
            #                        'G_rand_dis': loss_G_rand_dis.item(),
            #                        'G_recon_rand_dis': loss_G_recon_rand_dis.item(),
            #                        'G_rand_dis_acc': infogan_rand_acc,
            #                        'G_recon_rand_dis_acc': infogan_recon_rand_acc
            #                        })
            loss_dis = (loss_G_dis + loss_G_inv_dis + loss_G_rand_dis + loss_G_recon_rand_dis)
            dis_acc = (infogan_acc + infogan_inv_acc + infogan_rand_acc + infogan_recon_rand_acc) / 4.
            loss_G += args.lambda_dis * loss_dis
            current_losses.update({
                'dis': loss_dis.item(),
                'dis_acc': dis_acc
            })

        # FR loss, netFR must not be fixed
        loss_G_FR_total = 0
        cnt_G_FR = 0.

        if args.train_M:
            id_fake_G, fake_feat = model_dict['FR'](fake_aligned)
            loss_G_FR = -criterion_dict['FR'](id_fake_G, label.to(args.device))
            # current_losses['G_FR'] = loss_G_FR.item()

            id_rand_fake_G, rand_fake_feat = model_dict['FR'](rand_fake_aligned)
            loss_G_FR_rand = -criterion_dict['FR'](id_rand_fake_G, label.to(args.device))
            # current_losses['G_FR_rand'] = loss_G_FR_rand.item()

            loss_G_FR_total += loss_G_FR + loss_G_FR_rand
            cnt_G_FR += 2

        if args.feature_loss == 'cos':
            FR_cos_sim_target = torch.empty(size=(batch_size, 1), dtype=torch.float32, device=args.device)
            FR_cos_sim_target.fill_(-1.)

        if args.lambda_Feat:
            if args.feature_loss == 'cos':
                loss_G_feat = criterion_dict['Feat'](fake_feat, rand_fake_feat, target=FR_cos_sim_target)
            else:
                loss_G_feat = -criterion_dict['Feat'](fake_feat, rand_fake_feat)
            current_losses['G_feat'] = loss_G_feat.item()
            loss_G += args.lambda_Feat * loss_G_feat


        if args.lambda_G_recon:
            id_recon_G, recon_feat = model_dict['FR'](recon_aligned)
            if args.lambda_FR_WR:
                id_rand_recon_G, rand_recon_feat = model_dict['FR'](rand_recon_aligned)

            if args.lambda_recon_Feat:
                if args.feature_loss == 'cos':
                    loss_G_recon_feat = criterion_dict['Feat'](recon_feat, rand_recon_feat, target=FR_cos_sim_target)
                else:
                    loss_G_recon_feat = -criterion_dict['Feat'](recon_feat, rand_recon_feat)
                current_losses['G_recon_feat'] = loss_G_recon_feat.item()
                loss_G += args.lambda_recon_Feat * loss_G_recon_feat

            if args.lambda_false_recon_diff:
                if args.feature_loss == 'cos':
                    loss_G_false_recon_feat =criterion_dict['Feat'](fake_feat, rand_recon_feat, target=FR_cos_sim_target)
                else:
                    loss_G_false_recon_feat =-criterion_dict['Feat'](fake_feat, rand_recon_feat)
                current_losses['G_false_recon_feat'] = loss_G_false_recon_feat.item()
                loss_G += args.lambda_false_recon_diff * loss_G_false_recon_feat

            if args.recon_FR:
                loss_G_FR_recon = criterion_dict['FR'](id_recon_G, label.to(args.device))
                # current_losses['G_FR_recon'] = loss_G_FR_recon.item()
                if args.lambda_FR_WR:
                    loss_G_FR_rand_recon = -criterion_dict['FR'](id_rand_recon_G, label.to(args.device))
                else:
                    loss_G_FR_rand_recon = 0.
                # current_losses['G_FR_rand_recon'] = loss_G_FR_rand_recon.item()
                loss_G_FR_total += loss_G_FR_recon + args.lambda_FR_WR * loss_G_FR_rand_recon
                cnt_G_FR += 1. + args.lambda_FR_WR

        loss_G_FR_avg = loss_G_FR_total / cnt_G_FR

        loss_G += args.lambda_FR * loss_G_FR_avg
        current_losses['G_FR'] = loss_G_FR_avg.item()


        # loss_L1 = 0
        # cnt_loss_L1 = 0
        if args.lambda_L1 > 0:
            loss_G_L1 = criterion_dict['L1'](fake, img.cuda())
            current_losses['L1'] = loss_G_L1.item()
            # loss_L1 += loss_G_L1.item()
            # cnt_loss_L1 += 1
            loss_G += args.lambda_L1 * loss_G_L1

        if args.lambda_rand_L1 > 0:
            loss_G_rand_L1 = criterion_dict['L1'](rand_fake, img.cuda())
            current_losses['rand_L1'] = loss_G_rand_L1.item()
            # loss_L1 += loss_G_rand_L1.item()
            # cnt_loss_L1 += 1
            loss_G += args.lambda_rand_L1 * loss_G_rand_L1

        if args.lambda_rand_recon_L1 > 0:
            loss_G_rand_recon_L1 = criterion_dict['L1'](rand_recon, img.cuda())
            current_losses['wrong_recon_L1'] = loss_G_rand_recon_L1.item()
            # loss_L1 += loss_G_rand_recon_L1.item()
            # cnt_loss_L1 += 1
            loss_G += args.lambda_rand_recon_L1 * loss_G_rand_recon_L1

        # current_losses['L1'] = loss_L1 / float(cnt_loss_L1)

        if args.lambda_G_recon > 0:
            loss_G_recon = criterion_dict['L1'](recon, img.cuda())
            loss_G += args.lambda_G_recon * loss_G_recon
            current_losses['recon'] = loss_G_recon.item()

        if args.lambda_G_rand_recon > 0:
            if args.use_minus_one:
                inv_rand_z = rand_z * -1
            else:
                inv_rand_z = 1.0 - rand_z
            rand_fake_recon = model_dict['G'](rand_fake, inv_rand_z)
            loss_G_rand_recon = criterion_dict['L1'](rand_fake_recon, img.cuda())
            loss_G += args.lambda_G_rand_recon * loss_G_rand_recon
            current_losses['another_recon'] = loss_G_rand_recon.item()

        current_losses['G'] = loss_G.item()

        # G backward and optimizer steps
        loss_G.backward()
        optimizer_dict['G'].step()

        # -------------------- LOGGING PART --------------------
        if i % args.print_loss_freq == 0:
            t = (time.time() - iter_start_time) / batch_size
            visualizer.print_current_losses(epoch, i, current_losses, t, t_data)
            if args.display_id > 0 and i % args.plot_loss_freq == 0:
                visualizer.plot_current_losses(epoch, float(i) / len(train_loader), args, current_losses)
            if args.print_gradient:
                for net_name, net in model_dict.items():
                    # if net_name != 'Q':
                    #     continue
                    if isinstance(net, list):
                        continue
                    print(('================ NET %s ================' % net_name))
                    for name, param in net.named_parameters():
                        print_param_info(name, param, print_std=True)

        if i % args.visdom_visual_freq == 0:
            save_result = i % args.update_html_freq == 0

            current_visuals = OrderedDict()
            current_visuals['real'] = img.detach()
            current_visuals['fake'] = fake.detach()
            current_visuals['rand_fake'] = rand_fake.detach()
            if args.lambda_G_recon:
                current_visuals['recon'] = recon.detach()
                current_visuals['rand_recon'] = rand_recon.detach()
            if args.lambda_G_rand_recon > 0:
                current_visuals['rand_fake_recon'] = rand_fake_recon.detach()
            current_visuals['real_aligned'] = real_aligned.detach()
            current_visuals['fake_aligned'] = fake_aligned.detach()
            current_visuals['rand_fake_aligned'] = rand_fake_aligned.detach()
            if args.lambda_G_recon:
                current_visuals['recon_aligned'] = recon_aligned.detach()
                current_visuals['rand_recon_aligned'] = rand_recon_aligned.detach()

            try:
                with time_limit(60):
                    visualizer.display_current_results(current_visuals, epoch, save_result, args)
            except TimeoutException:
                visualizer.logger.log('TIME OUT visualizer.display_current_results epoch:{} iter:{}. Change display_id to -1'.format(epoch, i))
                args.display_id = -1

        if (i + 1) % args.save_iter_freq == 0:
            save_model(epoch, model_dict, optimizer_dict, args, iter=i)
            if args.display_id > 0:
                visualizer.vis.save([args.name])
                visualizer.overview_vis.save(['overview'])

        if (i + 1) % args.html_iter_freq == 0:
            test(test_loader, model_dict, criterion_dict, visualizer, epoch, args, fixed_z, fixed_rand_z, i)

        if (i + 1) % args.print_loss_freq == 0:
            iter_data_time = time.time()



def test(test_loader, model_dict, criterion_dict, visualizer, epoch, args, fixed_z, fixed_rand_z, iter=0):
    # TODO: whether to use eval mode
    import glob
    from utils import util
    import cv2
    image_folder = '/p300/dataset/face_anonymization_video/images'
    save_path = '/p300/FaceChange/user_study/video1'
    crop_txt = '/p300/project/mtcnn-pytorch/face_anonymization_video_detection.txt'
    with open(crop_txt) as f:
        images = f.readlines()

    tfm = transforms.Compose([
        transforms.Resize((args.imageSize, args.imageSize), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    fixed_z, fixed_dis_target, fixed_rand_z, fixed_rand_dis_target = generate_code(args.passwd_length,
                                                                                   args.batch_size,
                                                                                   args.device, inv=False)

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(images)):
            img_stat = None
            for img, _, _, _ in test_loader:
                img_stat = img
                break

            data = data.split(' ')
            img_path, bboxes = data[0], data[1:]
            img_path = os.path.join(image_folder, img_path)
            img = Image.open(img_path).convert('RGB')
            img_fake = img.copy()
            img_rand_fake = img.copy()
            img_recon = img.copy()
            img_rand_recon = img.copy()
            img_another_recon = img.copy()
            for i in range(len(bboxes) // 4):
                x1, y1, x2, y2 = bboxes[i * 4:(i + 1) * 4]
                x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
                face_img = img.crop((x1, y1, x2, y2))
                width, height = face_img.size
                face_img = tfm(face_img)
                img_stat[0] = face_img

                # fixed_z, fixed_dis_target, fixed_rand_z, fixed_rand_dis_target = generate_code(args.passwd_length,
                #                                                                                args.batch_size,
                #                                                                                args.device, inv=False)

                if args.lambda_dis > 0:
                    # concated_img = torch.cat((img, fixed_z.cpu().view(args.test_size, -1, 1, 1).repeat(1, 1, img.size(2), img.size(3))),
                    #                          1)
                    fake = model_dict['G'](img_stat, fixed_z.cpu())
                    rand_fake = model_dict['G'](img_stat, fixed_rand_z)
                    # concated_fake = torch.cat((fake, (1.0-fixed_z).view(args.test_size, -1, 1, 1).repeat(1, 1, img.size(2), img.size(3))),
                    #                           1)
                else:
                    fake = model_dict['G'](img_stat)

                current_visuals = OrderedDict()
                current_visuals['real'] = img_stat[:1]
                current_visuals['fake'] = fake[:1]
                current_visuals['rand_fake'] = rand_fake[:1]
                if args.lambda_G_recon:
                    assert args.lambda_dis > 0

                    if args.use_minus_one:
                        inv_z = fixed_z * -1
                    else:
                        inv_z = 1.0 - fixed_z

                    recon = model_dict['G'](fake, inv_z)
                    rand_recon = model_dict['G'](fake, fixed_rand_z)

                    if args.use_minus_one:
                        inv_rand_z = fixed_rand_z * -1
                    else:
                        inv_rand_z = 1.0 - fixed_rand_z
                    another_recon = model_dict['G'](rand_fake, inv_rand_z)
                    # else:
                    #     recon = model_dict['G'](fake)
                    current_visuals['recon'] = recon[:1]
                    current_visuals['rand_recon'] = rand_recon[:1]
                    current_visuals['another_recon'] = another_recon[:1]

                for label in ['fake', 'rand_fake', 'recon', 'rand_recon', 'another_recon']:
                    # import pdb
                    # pdb.set_trace()
                    vars()['img_'+label].paste(Image.fromarray(cv2.resize(util.tensor2im_all(current_visuals[label])[0], (width, height))), (x1, y1, x2, y2))

                    save_img_name = img_path.split('/')[-1]
                    split = save_img_name.split('.')
                    save_img_name = split[0] + '_' + label + '.' + split[1]
                    vars()['img_' + label].save(osp.join(save_path, save_img_name))

if __name__ == '__main__':
    main()
