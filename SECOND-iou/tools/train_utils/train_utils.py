import torch
import os
import glob
import tqdm
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils
from pcdet.utils import self_training_utils
from pcdet.config import cfg
import numpy as np


def visualize_boxes_batch(batch):
    import visualize_utils as vis
    import mayavi.mlab as mlab
    for b_idx in range(batch['batch_size']):
        points = batch['points'][batch['points'][:, 0] == b_idx][:, 1:]

        if 'debug' not in batch:
            vis.draw_scenes(points, ref_boxes=batch['gt_boxes'][b_idx, :, :7],
                            scores=batch['scores'][b_idx])
        else:
            vis.draw_scenes(points, ref_boxes=batch['gt_boxes'][b_idx, :, :7],
                            gt_boxes=batch['debug'][b_idx]['gt_boxes_lidar'],
                            scores=batch['scores'][b_idx])
        mlab.show(stop=True)


def train_one_epoch(model, optimizer, train_loader, model_func,prototype, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    # ps_bbox_meter = common_utils.AverageMeter()
    # ignore_ps_bbox_meter = common_utils.AverageMeter()

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()


        
        loss, tb_dict, disp_dict,prototype = model_func(model, batch, prototype)


        
        loss.backward()
        

        # pos_pseudo_bbox = target_batch['pos_ps_bbox'].mean().item()
        # ign_pseudo_bbox = target_batch['ign_ps_bbox'].mean().item()
        # ps_bbox_meter.update(pos_pseudo_bbox)
        # ignore_ps_bbox_meter.update(ign_pseudo_bbox)


        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter,prototype


def train_model(model, optimizer, train_loader, target_loader,prototype, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ps_label_dir,
                source_sampler=None, target_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None):
    accumulated_iter = start_iter

    # ps_pkl = self_training_utils.check_already_exsit_pseudo_label(ps_label_dir, start_epoch)
    # if ps_pkl is not None:
    #     logger.info('==> Loading pseudo labels from {}'.format(ps_pkl))



    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:

            
            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # # update pseudo label
            # if (cur_epoch in cfg.SELF_TRAIN.UPDATE_PSEUDO_LABEL) or \
            #         ((cur_epoch % cfg.SELF_TRAIN.UPDATE_PSEUDO_LABEL_INTERVAL == 0)
            #          and cur_epoch != 0):
            #     train_loader.dataset.eval()
            #     self_training_utils.save_pseudo_label_epoch(
            #         model, target_loader, rank, prototype,
            #         leave_pbar=True, ps_label_dir=ps_label_dir, cur_epoch=cur_epoch
            #     )
            #     train_loader.dataset.train()


            accumulated_iter,prototype = train_one_epoch(
                model, optimizer, train_loader, model_func,prototype,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter
            )

            

            proto_save_dir = str(ckpt_save_dir)[:-4] + 'proto'
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                state = checkpoint_state(model, optimizer, trained_epoch, accumulated_iter)

                save_checkpoint(state, filename=ckpt_name)

                pro_name = proto_save_dir + '/prototype_%d.npy' % trained_epoch

                np.save(pro_name,prototype.detach().cpu())


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
