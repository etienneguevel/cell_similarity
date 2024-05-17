import os
from dinov2.models import vision_transformer as dino_vits 

def make_encoder(args):
    if args.type == "dinov2":
        model = build_dinov2(args)
    else:
        raise RuntimeError(f"model type {args.type} has not been implemented yet.")
    
    return model

def build_dinov2(args, img_size=224):

    vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
    )
    model = dino_vits.__dict__[args.arch](**vit_kwargs)
    # if args.load_path is not None:
    #     assert os.path.exists(args.load_path)
    #     assert args.load_path.endwith(".pt", ".pth")
    #     try:
    #         model.load_state_dict(args.load_path)
    #     except Exception as e:
    #         raise RuntimeError(f"can not load the model from path {args.load_path}") from e

    return model

