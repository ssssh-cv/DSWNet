default_scope = 'mmagic'
experiment_name = 'testsfw'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

model = dict(
    type='BaseUWModel',
    generator=dict(type='SFWnet'),
    pixel_loss=dict(type='L1Loss', loss_weight=4.0, reduction='mean'),
    ssim_loss=dict(type='SSIMLoss', loss_weight=3.0),
    contrastive_loss=dict(
        type='ContrastiveLoss1',
        layer_weights=dict({
            '1': 0.03125,
            '6': 0.0625,
            '11': 0.125,
            '20': 0.25,
            '29': 1.0
        }),
        vgg_type='vgg19',
        perceptual_weight=0.15,
        norm_img=False),
    train_cfg=dict(),
    test_cfg=dict(metrics=[
        'PSNR',
        'SSIM',
    ]),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[
            0.0,
            0.0,
            0.0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ]))

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        out_dir='./work_dirs',
        by_epoch=False,
        max_keep_ckpts=10,
        save_best='PSNR',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'))

log_level = 'INFO'

log_processor = dict(type='LogProcessor', window_size=100, by_epoch=False)

# load_from = './work_dirs/fhw-origin/20250914_183131/best_PSNR_iter_12000.pth'

resume = False

vis_backends = [
    dict(type='LocalVisBackend'),
]

visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    fn_key='gt_path',
    img_keys=[
        'pred_img',
    ],
    bgr2rgb=True)

custom_hooks = [
    dict(type='BasicVisualizationHook', interval=1),
]

test_pipeline = [
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
    dict(type='PackInputs'),
]

uw_data_root = './datasets'

# for_s1000 = '/datafile/lcf_data/new_uw_testing/for_s1000'

uw_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='real', task_name='test'),
        data_root='E:/experiment/data256',
        data_prefix=dict(img='u45', gt='u45'),
        pipeline=[
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
            dict(type='PackInputs'),
        ]))

uw_evaluator = dict(
    type='Evaluator', metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])

test_cfg = dict(type='MultiTestLoop')

test_dataloader = [
    dict(
        num_workers=4,
        persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='BasicImageDataset',
            metainfo=dict(dataset_type='real', task_name='test'),
            data_root='./datasets',
            data_prefix=dict(img='c60', gt='c60'),
            pipeline=[
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
                dict(type='PackInputs'),
            ])),
]

test_evaluator = [
    dict(type='Evaluator', metrics=[
        dict(type='PSNR'),
        dict(type='SSIM'),
    ]),
]

launcher = 'none'
