class BaseParams:
    project = 'NERV'  # project name for wandb logging

    # training settings
    gpus = 1
    ddp_unused_params = False
    max_epochs = 100
    san_check_val_step = 2  # to verify code correctness
    print_iter = 50
    save_interval = 1.0  # save every (num_epoch_iters * save_interval) iters
    eval_interval = 1  # should be int, number of epochs between each eval
    save_epoch_end = False  # save ckp at the end of every epoch

    # optimizer settings
    optimizer = 'Adam'
    lr = 1e-3
    weight_decay = 0.0
    clip_grad = -1

    # data settings
    data_root = ''
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size * 2
    num_workers = 8
    grad_accum_steps = 1  # gradient accumulation

    # loss configs
    # we need to have `xxx_loss` as a key in the returned dict from the
    # `calc_train_loss` function in model
    # xxx_loss_w = 1.

    # we support changing some variables values with epoch
    # begin with 1., change to 10. at 40th epoch and then 100. at 80th epoch
    # var1 = 1.
    # var1_all = [1., 10., 100.]
    # var1_t = [40, 80]

    def get(self, key, value=None):
        """`get` function similar to dict."""
        if hasattr(self, key):
            return getattr(self, key)
        return value

    def to_dict(self):
        all_vars = [var for var in dir(self) if not var.startswith('__')]
        all_vars.remove('to_dict')
        all_vars.remove('from_dict')
        return {var: getattr(self, var) for var in all_vars}

    @staticmethod
    def from_dict(params_dict):
        params = BaseParams()
        for k, v in params_dict.items():
            setattr(params, k, v)
        return params
