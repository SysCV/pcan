_base_ = './segtrack-frcnn_r50_fpn_12e_bdd10k_fixed.py'
# model settings
model = dict(
    type='EMQuasiDenseMaskRCNNRefine',
    roi_head=dict(
        type='QuasiDenseSegRoIHeadRefine',
        double_train=False,
        mask_head=dict(type='FCNMaskHeadPlus'),
        refine_head=dict(
            type='EMMatchHeadPlus',
            num_convs=4,
            in_channels=256,
            conv_kernel_size=3,
            conv_out_channels=256,
            upsample_method='deconv',
            upsample_ratio=2,
            num_classes=8,
            pos_proto_num=10, #10
            neg_proto_num=10, #10
            stage_num=6,
            conv_cfg=None,
            norm_cfg=None,
            mask_thr_binary=0.5,
            match_score_thr=0.5,
            with_mask_ref=False,
            with_mask_key=True,
            with_dilation=False,
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0))),
    tracker=dict(type='QuasiDenseSegFeatEmbedTracker'))
    
load_from = './ckpts/segtrack-fixed-new.pth'
