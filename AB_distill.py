import argparse
import logging
import torch
import time
import numpy as np
from pathlib import Path
from threading import Thread

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import test
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import select_device, intersect_dicts, ModelEMA

from models.distillation import AB_Distill_Normal2Tiny, AB_Distill_Normal2Tiny2

logger = logging.getLogger(__name__)


def distill(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info('Start distilling')
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, epochs_distill, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.epochs_distill, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights_distill'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last_distill = wdir / 'last_distill.pt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    # Teacher
    ckpt = torch.load(weights, map_location=device)
    model_teacher = Model(ckpt['model'].yaml, ch=3, nc=nc).to(device)  # create teacher
    state_dict = ckpt['model'].float().state_dict()
    state_dict = intersect_dicts(state_dict, model_teacher.state_dict()) # intersect
    model_teacher.load_state_dict(state_dict, strict=False)  # load
    logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model_teacher.state_dict()), weights))  # report

    # Student
    model_student = Model(opt.cfg, ch=3, nc=nc).to(device)  # create student

    # Distill Module
    # distill_module = AB_Distill_Normal2Tiny(model_teacher, model_student, batch_size, 1.0).to(device)
    distill_module = AB_Distill_Normal2Tiny(model_teacher, model_student, batch_size, 1.0).to(device)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay_distill'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay_distill = {hyp['weight_decay_distill']}")

    # optimizer = optim.SGD([{'params': model_student.model[:11].parameters()}, \
    #     {'params': distill_module.Connectors.parameters()}], lr=hyp['lr0_distill'], momentum=hyp['momentum_distill'], nesterov=True, weight_decay=hyp['weight_decay_distill'])

    optimizer = optim.SGD([{'params': model_student.model[:7].parameters()}, \
        {'params': distill_module.Connectors.parameters()}], lr=hyp['lr0_distill'], momentum=hyp['momentum_distill'], nesterov=True, weight_decay=hyp['weight_decay_distill'])

    # optimizer = optim.Adam([{'params': model_student.model[:11].parameters()}, \
    #     {'params': distill_module.Connectors.parameters()}], lr=hyp['lr0_distill'])

    # Logging
    if wandb and wandb.run is None:
        opt.hyp = hyp  # add hyperparameters
        wandb_run = wandb.init(config=opt, resume="allow",
                               project='YOLOv5_Distill' if opt.project == 'runs/AB_distill' else Path(opt.project).stem,
                               name=save_dir.stem,
                               id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)
    loggers = {'wandb': wandb}  # loggers dict

    # Image sizes
    gs = int(model_student.stride.max())  # grid size (max stride)
    nl = model_student.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Not resume
    labels = np.concatenate(dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes
    # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
    # model._initialize_biases(cf.to(device))

    # Anchors
    if not opt.noautoanchor:
        check_anchors(dataset, model=model_student, thr=hyp['anchor_t'], imgsz=imgsz)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    model_student.nc = nc  # attach number of classes to model
    model_student.hyp = hyp  # attach hyperparameters to model
    model_student.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model_student.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model_student.names = names
    
    model_teacher.nc = nc
    model_teacher.hyp = hyp
    model_teacher.gr = 1.0
    model_teacher.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model_student.names = names

    # Distillation loop -------------------------------------------------------------------------------------------------------------------------------------
    t0 = time.time()
    scaler = amp.GradScaler(enabled=cuda)
    logger.info(f'Image sizes {imgsz} train\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting distilling for {epochs_distill} epochs...')
    start_epoch_distill = 0
    for epoch in range(start_epoch_distill, epochs_distill):
        model_teacher.eval()
        model_student.train()
        distill_module.train()

        mloss = torch.zeros(7, device=device)  # mean losses
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 11) % ('Epoch', 'gpu_mem', 'loss', 'loss_g1', 'loss_g2', 'loss_g3', 'sim_g1', 'sim_g2', 'sim_g3', 'targets', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar

        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # Forward
            with amp.autocast(enabled=cuda):
                loss, loss_g1, loss_g2, loss_g3, sim_g1, sim_g2, sim_g3 = distill_module(imgs)  # forward
            
            # Backward
            scaler.scale(loss).backward()

            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()

            # Print
            loss_items = torch.Tensor([loss, loss_g1, loss_g2, loss_g3, sim_g1, sim_g2, sim_g3]).detach().to(device)
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 9) % (
                '%g/%g' % (epoch, epochs_distill - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

        lr = [x['lr'] for x in optimizer.param_groups]

        # Log
        tags = ['distill/loss', 'distill/loss_g1', 'distill/loss_g2', 'distill/loss_g3', 'distill/loss_sim1', 'distill/loss_sim2', 'distill/loss_sim3', 'x/lr_distill']  # params
        for x, tag in zip(list(mloss)+lr, tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard
            if wandb:
                wandb.log({tag: x})  # W&B

        # Save model
        ckpt = {'epoch': 0,
                'model': model_student.eval(),
                'optimizer': None,
                'wandb_id': wandb_run.id if wandb else None}

        # Save last and delete
        torch.save(ckpt, last_distill)
        del ckpt
    # End distillation loop ---------------------------------------------------------------------------------------------------------------------------------
    if epochs == 0: # Distillation only
        if last_distill.exists():
            strip_optimizer(last_distill)  # strip optimizers
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path for teacher network')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path for student network')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.distill.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--epochs-distill', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts, i.e. final trained model')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/AB_distill', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    opt = parser.parse_args()

    opt.world_size = 1
    opt.global_rank = -1
    set_logging()

    # Setup path
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.weights), '--weights must be specified for teacher network'
    assert len(opt.cfg), '--cfg must be specified for student network'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # Device
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Distill and train
    import wandb
    logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
    tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    distill(hyp, opt, device, tb_writer, wandb)

    