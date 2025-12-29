_base_ = [
    '../_base_/default_runtime.py',
]

experiment_name = 'sfw'

work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

model = dict(
    type='BaseUWModel',
    generator=dict(type='SFWnet'),
    pixel_loss=dict(type='L1Loss', loss_weight=4.0, reduction='mean'),

    ssim_loss=dict(type='SSIMLoss', loss_weight=3.0),
    contrastive_loss=dict(
        type='ContrastiveLoss1',
        layer_weights={
            '1': 1.0 / 32,
            '6': 1.0 / 16,
            '11': 1.0 / 8,
            '20': 1.0 / 4,
            '29': 1.,
        },
        vgg_type='vgg19',
        perceptual_weight=0.15,
        norm_img=False),
    train_cfg=dict(),
    test_cfg=dict(metrics=['PSNR', 'SSIM']),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.]))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='Resize',
        keys=['img', 'gt'],
        scale=(256, 256)),
    dict(type='SetValues', dictionary=dict(scale=1)),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='vertical'),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb'),
    dict(
        type='Resize',
        keys=['img', 'gt'],
        scale=(256, 256)),
    dict(type='PackInputs')
]


train_dataloader = dict(
    num_workers=4,
    batch_size=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='mix', task_name='train'),

        data_root='put your data path',

        data_prefix=dict(img='lq', gt='gt'),
        pipeline=train_pipeline))

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='real', task_name='val'),

        data_root='put your data path',

        data_prefix=dict(img='lq', gt='gt'),
        pipeline=val_pipeline))

val_evaluator = dict(
    type='Evaluator',
    metrics=[dict(type='PSNR'),
             dict(type='SSIM')])

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=500)
val_cfg = dict(type='MultiValLoop')

optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-3, betas=(0.9, 0.999)))


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=500,
        max_keep_ckpts=1,
        save_best='SSIM',
        rule='greater',
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['input', 'pred_img', 'gt_img'],
    bgr2rgb=True)
custom_hooks = [dict(type='BasicVisualizationHook', interval=3)]
