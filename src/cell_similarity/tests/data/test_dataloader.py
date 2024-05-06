import os
from functools import partial
from pathlib import Path
from cell_similarity.data.datasets import ImageDataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
from dinov2.data import SamplerType, make_data_loader, make_dataset

def test():
    
    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    
    path_dataset_test = os.path.join(os.getcwd(), 'dataset_test')
    dataset = ImageDataset(root=path_dataset_test, transform=data_transform)

    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        sampler_type=sampler_type,
        sampler_advance=0,  
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    for i in data_loader:
        assert len(i) == cfg.train.batch_size_per_gpu
        break

if __name__ == '__main__':
    test()
