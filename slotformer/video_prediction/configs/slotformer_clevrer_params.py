from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 2  # 2 GPUs should also be good
    max_epochs = 30  # ~450k steps
    save_interval = 0.25  # save every 0.25 epoch
    eval_interval = 4  # evaluate every 4 epochs
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 5  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 2e-4
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps
    # no weight decay, no gradient clipping

    # data settings
    dataset = 'clevrer_slots'
    data_root = '/cs/data/people/hnam16/data/clevrer_for_savi'
    slots_root = '/cs/data/people/hnam16/data/modified_extraction/clevrer_savi_reproduced.pkl'
    n_sample_frames = 6 + 10  # 6 burn-in, 10 rollout
    frame_offset = 2  # subsample the video by a factor of 2
    load_img = True
    filter_enter = True  # only train on valid sequences
    train_batch_size = 32
    val_batch_size = train_batch_size * 2
    num_workers = 8  # 8 causes hanging due to multiprocessing deadlock with CUDA

    # model configs
    model = 'SlotFormer'
    resolution = (64, 64)
    input_frames = 6  # burn-in frames

    num_slots = 7
    slot_size = 128
    slot_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
    )

    # Rollouter
    rollout_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
        history_len=input_frames,
        t_pe='sin',  # sine temporal P.E.
        slots_pe='',  # no slots P.E.
        # Transformer-related configs
        d_model=slot_size * 2,
        num_layers=4,
        num_heads=8,
        ffn_dim=slot_size * 2 * 4,
        norm_first=True,
    )

    # CNN Decoder
    dec_dict = dict(
        dec_channels=(128, 64, 64, 64, 64),
        dec_resolution=(8, 8),
        dec_ks=5,
        dec_norm='',
        dec_ckp_path='/cs/data/people/hnam16/savi_pretrained/clevrer_savi_reproduce_20260109_102641_LR0.0001/savi/epoch/model_8.pth',
    )

    # loss configs
    loss_dict = dict(
        rollout_len=n_sample_frames - rollout_dict['history_len'],
        use_img_recon_loss=False,  # important for predicted image quality
    )

    slot_recon_loss_w = 1.
    img_recon_loss_w = 0.
