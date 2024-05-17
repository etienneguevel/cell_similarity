import os 
from omegaconf import OmegaConf

from cell_similarity.configs import default_config

def write_config(cfg, output_dir, name="config.yaml"):
    OmegaConf.to_yaml(cfg)
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path

def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir)

    default_cfg = OmegaConf.create(default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg)
    return cfg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    write_config(cfg, args.output_dir)
    return cfg
