import os
import time
import wandb
import numpy as np
from tqdm import tqdm
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from nerv.utils.io import check_file_exist, mkdir_or_exist
from nerv.utils.misc import AverageMeter, MeanMetric
from nerv.utils.tensor import ddp_all_gather
from nerv.utils.conversion import is_list_of
from nerv.training.lr import get_lr
from nerv.training.model import BaseModel
from nerv.training.params import BaseParams
from nerv.training.datamodule import BaseDataModule
from nerv.models.utils import filter_wd_parameters


class BaseMethod(nn.Module):
    """Base method for training a model.

    Args:
        model (`BaseModel`): the network to train. See `model.py`.
        datamodule (`BaseDataModule`): the datamodule with train/val loader.
            See `datamodule.py`.
        params (`BaseParams`): the parameter settings. See `params.py`.
        ckp_path (str): root to save the checkpoints.
        local_rank (int, optional): used in DDP training. Default: 0.
        use_ddp (bool, optional): whether use DDP training. Default: False.
        use_fp16 (bool, optional): whether use FP16 mixed precision training.
            Default: False.

    Member Variables:
        # model
        `self.model`: as `model`.

        # optimization
        `self.optimizer`: optimizer.
        `self.grad_scaler`: used in FP16 mixed precision training.
        `self.scheduler`: lr scheduler.
        `self.scheduler_method`: determine how to adjust lr, 'step' or 'epoch'.
        `self.clip_grad`: clipping gradient value. Not clipping if <= 0.

        # data related
        `self.train_loader`: `datamodule.train_loader`.
        `self.iter_train_loader`: `iter(self.train_loader)`, used for saving
            data_loader's state in checkpoint.
        `self.val_loader`: `datamodule.val_loader`.

        # statistics
        `self.it`: total training iterations.
        `self.epoch`: total training epochs.
        `self.print_iter`: interval between print/log training statistics.
        `self.save_iter`: interval between saving checkpoint.
        `self.save_epoch_end`: whether to save ckp at the end of every epoch.
        `self.epoch_it`: iteration number in one epoch.
        `self.stats_dict`: accumulate values. Can be used for avg.

        # all parameter settings
        `self.params`: as `params`.
        `self.ckp_path`: as `ckp_path`.
        `self.local_rank`: as `local_rank`.
        `self.grad_accum_steps`: gradient accumulation steps.
        `self.use_ddp`: as `use_ddp`.
        `self.use_fp16`: as `use_fp16`.
        `self.gpus`: number of GPUs available, should equal to `params.gpus`.
        `self.device`: device of current process, e.g. `cuda:0`.

    """

    def __init__(
        self,
        model: BaseModel,
        datamodule: BaseDataModule,
        params: BaseParams,
        ckp_path: str,
        local_rank: int = 0,
        use_ddp: bool = False,
        use_fp16: bool = False,
    ):
        super().__init__()

        # model & params
        self.model = model
        self.params = params

        # DDP training
        self.use_ddp = use_ddp
        self.local_rank = local_rank
        self._init_env()

        # FP16 mixed precision training
        self.use_fp16 = use_fp16
        self._init_amp()

        # data
        self.train_loader = datamodule.train_loader
        self.val_loader = datamodule.val_loader

        # training settings
        self.grad_accum_steps = params.get('grad_accum_steps', 1)
        self.ckp_path = ckp_path
        self._init_training()

    def _init_env(self):
        """Init training environment related settings.

        - `self.device`
        - DDP training
        """
        self.gpus = torch.cuda.device_count()
        assert self.gpus == self.params.gpus, 'GPU number not aligned'

        # DDP init
        if self.use_ddp:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(
                'nccl', init_method='env://', timeout=timedelta(hours=2))
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cuda')
            assert self.gpus == 1, 'Multi-GPU training should use DDP'
            assert self.local_rank == 0, \
                'local rank should be 0 in non DDP training'

        # model to device
        self.model = self.model.to(self.device)
        if self.use_ddp:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.params.ddp_unused_params)
        else:
            # this is only to enable calling `self.model.module`
            self.model = nn.parallel.DataParallel(
                self.model, device_ids=list(range(self.gpus)))

    def _init_amp(self):
        """Init FP16 mixed precision training related settings."""
        if not self.use_fp16:
            return

        self.grad_scaler = torch.cuda.amp.GradScaler()

    def _init_training(self):
        # training accumulator
        self.it, self.epoch = 0, 0
        self.max_epochs = self.params.max_epochs
        self.print_iter = self.params.print_iter
        self.save_iter = int(
            np.ceil(len(self.train_loader) * self.params.save_interval)) + 1
        self.eval_interval = self.params.eval_interval
        self.save_epoch_end = self.params.save_epoch_end

        if self.local_rank == 0:
            mkdir_or_exist(self.ckp_path)
            if self.save_epoch_end:
                self.epoch_ckp_path = os.path.join(self.ckp_path, 'epoch')
                mkdir_or_exist(self.epoch_ckp_path)

        # gradient clipping
        self.clip_grad = self.params.clip_grad

        # construct optimizer and lr_scheduler
        self._setup_optimizer()

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def fit(self, resume_from='', san_check_val_step=2):
        """Train the model.

        Args:
            resume_from (str, optional): pre-trained weight path.
                Default: ''.
            san_check_val_step (int, optional): run a few val steps to verify
                the model, logging, checkpointing etc implementations.
                Default: 2.
        """
        # automatically detect existing checkpoints
        self.load_ckp(ckp_path=resume_from)

        # run several val steps as sanity check
        if san_check_val_step > 0:
            self.validation_epoch(
                self.model.module, san_check_step=san_check_val_step)

        for _ in range(self.epoch, self.max_epochs):
            self.training_epoch()

    def training_epoch(self):
        """Train one epoch.

        - Calculate loss via `self.training_step`
        - Backward and perform one optimization step
        - Apply lr scheduler
        - Accumulate iter and epoch number
        """
        self._training_epoch_start()

        # use iter dataloader in order to save its state
        self.iter_train_loader = iter(self.train_loader)
        train_steps = (
            len(self.train_loader.sampler) -
            self.train_loader.sampler.real_counter(
                self.iter_train_loader)) // self.params.train_batch_size
        tqdm_desc = f'Train epoch {self.epoch}, rank {self.local_rank}'
        with tqdm(total=train_steps, desc=tqdm_desc) as t:
            t1 = time.time()
            for batch_idx, batch_data in enumerate(self.iter_train_loader):
                # torch.cuda.empty_cache()

                # data time
                t2 = time.time()
                data_time = t2 - t1

                # set the batch idx
                self.epoch_it = batch_idx
                batch_data = {
                    k: v.to(self.device)
                    for k, v in batch_data.items()
                }

                self._training_step_start()

                # model forward, loss computation, backward and optimize
                out_dict = self._training_step(batch_data)

                # forward time
                t1 = time.time()
                forward_time = t1 - t2
                out_dict['data_time'] = self._make_tensor(data_time)
                out_dict['forward_time'] = self._make_tensor(forward_time)

                # logging
                self._log_train(out_dict)

                self._training_step_end()

                t.set_postfix(loss=f"{out_dict['loss'].item():.4f}")
                t.update(1)

        self._training_epoch_end()

    def _loss_function(self, batch_data):
        """Compute and aggregate losses."""
        out_dict = self.model.module.loss_function(batch_data)
        assert 'loss' not in out_dict.keys()
        loss = self._make_tensor(0.)
        for loss_name, loss_value in out_dict.items():
            if not loss_name.endswith('_loss'):
                continue
            loss = loss + loss_value * eval(f'self.params.{loss_name}_w')
        out_dict['loss'] = loss
        return out_dict

    def _training_step_fp32(self, batch_data):
        """Loss backward and optimize in the normal FP32 setting."""
        out_dict = self._loss_function(batch_data)

        # normalize loss to account for batch accumulation
        loss = out_dict['loss'] / self.grad_accum_steps

        # BP
        loss.backward()

        # weights update
        if ((self.epoch_it + 1) % self.grad_accum_steps == 0) or \
                (self.epoch_it + 1 == len(self.iter_train_loader)):
            # gradient clipping
            if self.clip_grad > 0.:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.clip_grad)

            # optimize one step
            self.optimizer.step()
            self.optimizer.zero_grad()

        return out_dict

    def _training_step_fp16(self, batch_data):
        """Loss backward and optimize in FP16 mixed precision setting."""
        with torch.cuda.amp.autocast():
            out_dict = self._loss_function(batch_data)

            # normalize loss to account for batch accumulation
            loss = out_dict['loss'] / self.grad_accum_steps

        # scaled BP
        self.grad_scaler.scale(loss).backward()

        # weights update
        if ((self.epoch_it + 1) % self.grad_accum_steps == 0) or \
                (self.epoch_it + 1 == len(self.iter_train_loader)):
            # gradient clipping
            if self.clip_grad > 0.:
                self.grad_scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.clip_grad)

            # optimize one step
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()

        return out_dict

    def _training_step(self, batch_data):
        """Returns a dict containing 'loss' to apply optimizer on.
        The values in the dict will be logged to wandb.
        """
        if self.use_fp16:
            return self._training_step_fp16(batch_data)
        return self._training_step_fp32(batch_data)

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        # only log results for rank 0
        if self.local_rank != 0:
            return

        # accumulate results till print interval
        self._accumulate_stats(out_dict)

        if (self.epoch_it + 1) % self.print_iter != 0:
            return

        out_dict = {f'train/{k}': v.avg for k, v in self.stats_dict.items()}
        out_dict['train/epoch'] = self.epoch
        out_dict['train/it'] = self.it
        out_dict['train/lr'] = get_lr(self.optimizer)
        if self.use_fp16:
            out_dict['train/fp16_loss_scale'] = self.grad_scaler.get_scale()
        wandb.log(out_dict, step=self.it)
        self.stats_dict = None

    def _training_epoch_start(self):
        """Things to do at the beginning of every training epoch."""
        self.optimizer.zero_grad()
        self.train()
        self.stats_dict = None

        # update some values in self.params depending on epoch number
        all_vars = [
            var for var in dir(self.params) if not var.startswith('__')
            and not callable(getattr(self.params, var))
        ]
        for var in all_vars:
            if var.endswith('_t') and f'{var[:-2]}_all' in all_vars:
                var_name = var[:-2]
                var_t = getattr(self.params, var)
                var_all = getattr(self.params, f'{var_name}_all')
                counter = 0
                for t in var_t:
                    if self.epoch >= t:
                        counter += 1
                    else:
                        break
                new_var = var_all[counter]
                if getattr(self.params, var_name) == new_var:
                    continue
                setattr(self.params, var_name, new_var)
                print(f'Changing params.{var_name} to {new_var}')

        # call the same method for model
        self.model.module._training_epoch_start(method=self)

        print(f'>>> Training epoch {self.epoch} start, rank {self.local_rank}')

        # sync DDP training processes at the beginning of epoch
        if self.use_ddp:
            torch.distributed.barrier()

    def _training_step_start(self):
        """Things to do at the beginning of every training step."""
        # call the same method for model
        self.model.module._training_step_start(method=self)

    def _training_step_end(self):
        """Things to do at the end of every training step."""
        if self.scheduler_method == 'step':
            self.scheduler.step()
        self.it += 1

        # call the same method for model
        self.model.module._training_step_end(method=self)

        if (self.epoch_it + 1) % self.save_iter == 0:
            self.save_ckp(save_loader=True)
            # sync DDP training processes at the end of step
            if self.use_ddp:
                torch.distributed.barrier()

    def _training_epoch_end(self):
        """Things to do at the end of every training epoch."""
        if self.scheduler_method == 'epoch':
            self.scheduler.step()
        self.epoch += 1
        self.stats_dict = None

        # call the same method for model
        self.model.module._training_epoch_end(method=self)

        self.save_ckp(save_loader=False)
        if self.use_ddp:
            torch.distributed.barrier()

        # run one epoch of validation after each training epoch
        if (self.epoch + 1) % self.eval_interval == 0:
            self.validation_epoch(self.model.module)

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.

        According to a discussion, in DDP eval we should pass the inner module
            of DDPModel for eval to prevent hanging.
        See: https://discuss.pytorch.org/t/torch-distributed-barrier-hangs-in-ddp/114522.  # noqa
        """
        print(f'>>> Evaluating start, rank: {self.local_rank}')
        model.eval()
        self.stats_dict = None
        torch.cuda.empty_cache()

        tqdm_desc = f'Eval epoch {self.epoch}, rank {self.local_rank}'
        for batch_idx, batch_data in enumerate(
                tqdm(self.val_loader, desc=tqdm_desc)):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}

            out_dict = model.loss_function(batch_data)
            self._accumulate_stats(out_dict, test=True)

            # if it's actually in sanity check
            if san_check_step > 0 and batch_idx + 1 >= san_check_step:
                break

        # log eval statistics
        if san_check_step <= 0:
            # we explicitly follow keys order for DDP sync
            all_keys = sorted(list(self.stats_dict.keys()))
            out_dict = {
                f'val/{k}': self.stats_dict[k].compute().item()
                for k in all_keys
            }
            print(f'Eval epoch {self.epoch}, rank {self.local_rank} result:',
                  out_dict)
            if self.local_rank == 0:
                wandb.log(out_dict, step=self.it)
                print(f'Eval epoch {self.epoch}, rank {self.local_rank} log')

        self.stats_dict = None
        torch.cuda.empty_cache()
        print(f'>>> Evaluating end, rank: {self.local_rank}')

    def _setup_optimizer(self):
        """Construct optimizer and lr scheduler."""
        self.optimizer, (self.scheduler, self.scheduler_method) = \
            self._configure_optimizers()
        assert self.scheduler_method in ['step', 'epoch', '']
        if not self.scheduler_method:
            assert self.optimizer is None

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        lr = self.params.lr
        wd = self.params.weight_decay
        if wd > 0.:
            params_dict = filter_wd_parameters(self.model)
            params_list = [{
                'params': params_dict['no_decay'],
                'weight_decay': 0.,
            }, {
                'params': params_dict['decay'],
                'weight_decay': wd,
            }]
        else:
            params_list = filter(lambda p: p.requires_grad,
                                 self.model.parameters())

        # we use Adam by default
        if self.params.optimizer.lower() == 'adam':
            if wd > 0.:
                # use AdamW in weight_decay
                optimizer = optim.AdamW(params_list, lr=lr)
            else:
                optimizer = optim.Adam(params_list, lr=lr, weight_decay=0.)
        else:
            if self.params.optimizer.lower() == 'sgd':
                momentum = self.params.momentum if \
                    hasattr(self.params, 'momentum') else 0.9
                optimizer = optim.SGD(
                    params_list, lr=lr, momentum=momentum, weight_decay=wd)
            elif self.params.optimizer.lower() == 'rmsprop':
                optimizer = optim.RMSprop(params_list, lr=lr, weight_decay=wd)
            else:
                raise NotImplementedError(
                    f'unsupported optimizer {self.params.optimizer}')
        return optimizer, (None, '')

    def _make_tensor(self, x):
        """Convert `x` to torch.tensor on `self.device`.

        Usually used in DDP all_gather some non-torch variables.
        """
        return torch.tensor(x).to(self.device)

    @torch.no_grad()
    def _accumulate_stats(self, stats_dict, test=False):
        """Append stats in `stats_dict` to `self.stats_dict`.

        We assume that each value in stats_dict is a torch scalar.
        In training time, only average over device:0, while at test time,
            we need to gather metrics over all the device.
        """
        bs = stats_dict.pop('batch_size', 1)
        # we explicitly follow keys order for DDP sync
        all_keys = sorted(list(stats_dict.keys()))
        if self.stats_dict is None:
            meter = MeanMetric if test else AverageMeter
            self.stats_dict = {k: meter(device=self.device) for k in all_keys}
        for k in all_keys:
            v = stats_dict[k]
            item = self._make_tensor(v.item()) if test else v.item()
            self.stats_dict[k].update(item, bs)

    @torch.no_grad()
    def _gather_train_sampler_state(self):
        """Gather the state for self.train_loader across DDP."""
        sampler_state = self.train_loader.sampler.state_dict(
            self.iter_train_loader)
        indices = sampler_state['indices']  # a list of int
        counter = sampler_state['counter']  # an int
        assert is_list_of(indices, int) and isinstance(counter, int)

        # not in DDP, directly get one and return
        if not self.use_ddp:
            return {
                'rank0_train_sampler': {
                    'indices': indices,
                    'counter': counter
                },
            }

        # need to gather all rank dataloader in DDP
        # all_gather only supports tensor, so we convert them into tensor
        sampler_state = indices + [counter]
        sampler_state = self._make_tensor(sampler_state)
        gather_state = ddp_all_gather(sampler_state).cpu()
        indices, counter = gather_state[:, :-1], gather_state[:, -1]
        # lists of shape [world_size, N], [world_size]
        indices, counter = indices.tolist(), counter.tolist()
        assert len(indices) == len(counter) == dist.get_world_size()
        return {
            f'rank{i}_train_sampler': {
                'indices': indices[i],
                'counter': counter[i]
            }
            for i in range(len(counter))
        }

    @torch.no_grad()
    def save_ckp(self, save_loader=False, keep_num=5):
        """Save state_dict of all self.modules.

        The default save name is '{self.ckp_path}/model_{self.it}.pth'.
        """
        torch.cuda.empty_cache()

        # if at the middle of training, should save dataloader states
        # if in eval, no need to do so
        if save_loader:
            train_sampler_states = self._gather_train_sampler_state()

        # only rank 0 process save ckp
        if self.local_rank != 0:
            return

        # auto remove earlier ckps
        ckp_files = os.listdir(self.ckp_path)
        ckp_files = [ckp for ckp in ckp_files if ckp.endswith('.pth')]
        if keep_num > 0 and len(ckp_files) >= keep_num:
            ckp_files = sorted(
                ckp_files,
                key=lambda x: os.path.getmtime(os.path.join(self.ckp_path, x)))
            del_ckp = ckp_files[:-(keep_num - 1)]
            for x in del_ckp:
                os.remove(os.path.join(self.ckp_path, x))

        ckp_path = os.path.join(self.ckp_path, f'model_{self.it}.pth')
        ckp = {
            'state_dict': self.model.module.state_dict(),
            'opt_state_dict': self.optimizer.state_dict(),
            'it': self.it,
            'epoch': self.epoch,
        }
        if self.scheduler_method:
            ckp['scheduler_state_dict'] = self.scheduler.state_dict()
            ckp['scheduler_method'] = self.scheduler_method
        if self.use_fp16:
            ckp['grad_scaler'] = self.grad_scaler.state_dict()
        if save_loader:
            ckp.update(train_sampler_states)
        torch.save(ckp, ckp_path)
        print(f'INFO: saving checkpoint {ckp_path}')

        # we may want to save ckp at the end of every epoch
        # these ckps won't be restricted by `keep_num`
        if not save_loader and self.save_epoch_end:
            ckp_path = os.path.join(self.epoch_ckp_path,
                                    f'model_{self.epoch}.pth')
            torch.save(ckp, ckp_path)
            print(f'INFO: saving checkpoint {ckp_path}')

    @torch.no_grad()
    def load_ckp(self, ckp_path=None, auto_detect=True):
        """Load from checkpoint.

        Support automatic detection of existing checkpoints.
        Useful in SLURM preemption systems.
        """
        # automatically detect checkpoints
        if auto_detect and os.path.exists(self.ckp_path):
            ckp_files = os.listdir(self.ckp_path)
            ckp_files = [ckp for ckp in ckp_files if ckp.endswith('.pth')]
            if ckp_files:
                ckp_files = sorted(
                    ckp_files,
                    key=lambda x: os.path.getmtime(
                        os.path.join(self.ckp_path, x)))
                last_ckp = ckp_files[-1]
                print(f'INFO: automatically detect checkpoint {last_ckp}')
                ckp_path = os.path.join(self.ckp_path, last_ckp)

        if not ckp_path:
            return

        print(f'INFO: loading checkpoint {ckp_path}')
        check_file_exist(ckp_path)
        ckp = torch.load(ckp_path, map_location='cpu')
        self.it, self.epoch = ckp['it'], ckp['epoch']
        self.model.module.load_state_dict(ckp['state_dict'])
        self.optimizer.load_state_dict(ckp['opt_state_dict'])
        if self.scheduler_method:
            self.scheduler.load_state_dict(ckp['scheduler_state_dict'])
            self.scheduler_method = ckp['scheduler_method']
        if self.use_fp16:
            self.grad_scaler.load_state_dict(ckp['grad_scaler'])
        # should consider loading data sampler
        if 'rank0_train_sampler' in ckp.keys():
            print('INFO: loading train loader state')
            self.train_loader.sampler.load_state_dict(
                ckp[f'rank{self.local_rank}_train_sampler'])
