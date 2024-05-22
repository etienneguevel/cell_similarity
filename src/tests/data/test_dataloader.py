from cell_similarity.data import make_datasets, make_dataloaders

def test_dataloaders():
    path_dataset_test = '../resources/dataset_test'
    datasets = make_datasets(root=path_dataset_test)
    dataloaders = make_dataloaders(datasets=datasets, batch_size=32)

    assert len(dataloaders.keys()) == 3
    for batch, labels in dataloaders["train"]:
        break

    assert batch.shape[0] == 32
    assert labels.shape[0] == 32
    
