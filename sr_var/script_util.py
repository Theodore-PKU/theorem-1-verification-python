from models.unet import UNetModel, VarianceModel


def mean_model_defaults():
    """
    Defaults for mean model training.
    :return: a dict that contains parameters setting.
    """
    return dict(
        large_size=256,
        small_size=64,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions="16, 8",
        dropout=0,
        channel_mult="",
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=False,
        use_new_attention_order=False,
    )


def create_mean_model(
    large_size,
    small_size,
    in_channels,
    model_channels,
    out_channels,
    num_res_blocks,
    attention_resolutions,
    dropout,
    channel_mult,
    conv_resample,
    dims,
    use_checkpoint,
    use_fp16,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    resblock_updown,
    use_new_attention_order,
):
    _ = small_size
    if channel_mult == "":
        if large_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif large_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        else:
            raise ValueError(f"unsupported image size: {large_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return UNetModel(
        image_size=large_size,
        in_channels=in_channels,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        conv_resample=conv_resample,
        dims=dims,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def var_model_defaults():
    """
    Defaults for variance model training.
    :return: a dict that contains parameters setting.
    """
    return dict(
        large_size=256,
        small_size=64,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions="16, 8",
        dropout=0,
        channel_mult="",
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=False,
        use_new_attention_order=False,
        last_layer_type="none",
    )


def create_var_model(
        large_size,
        small_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout,
        channel_mult,
        conv_resample,
        dims,
        use_checkpoint,
        use_fp16,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        resblock_updown,
        use_new_attention_order,
        last_layer_type,
):
    _ = small_size
    if channel_mult == "":
        if large_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif large_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        else:
            raise ValueError(f"unsupported image size: {large_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return VarianceModel(
        image_size=large_size,
        in_channels=in_channels,
        last_layer_type=last_layer_type,
        model_channels=model_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        conv_resample=conv_resample,
        dims=dims,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )
