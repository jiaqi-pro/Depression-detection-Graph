necks=dict(
        type='TPN',
        in_channels=[256,512,1024,2048],
        out_channels=2048,
        spatial_modulation_config=dict(
            inplanes=[256,512,1024,2048],
            planes=2048,
        ),
        temporal_modulation_config=dict(
            scales=(8,8,8,8),
            param=dict(
                inplanes=[],
                planes=-1,
                downsample_scale=-1,
            )),
        upsampling_config=dict(
            scale=(1, 1, 1),
        ),
        downsampling_config=dict(
            scales=(1, 1, 1),
            param=dict(
                inplanes=-1,
                planes=-1,
                downsample_scale=-1,
            )),
        level_fusion_config=dict(
            in_channels=[1024, 1024,1024,1024],
            mid_channels=[1024, 1024,1024,1024],
            out_channels=2048,
            ds_scales=[(1, 1, 1), (1, 1, 1),(1, 1, 1),(1, 1, 1)],
        ),
        aux_head_config=dict(
            inplanes=-1,
            planes=174,
            loss_weight=0.5
        ),
    )
backbone=dict(
        type='ResNet',
        pretrained='modelzoo://resnet50',
        depth=50,
        nsegments=8,
        out_indices=(2, 3),
        tsm=True,
        bn_eval=False,
        partial_bn=False)

