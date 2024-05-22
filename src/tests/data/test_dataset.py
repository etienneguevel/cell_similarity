from cell_similarity.data import make_datasets

def test_dataset():
    datasets = make_datasets(root='../resources/dataset_test')
    assert len(datasets.keys()) == 3

if __name__ == '__main__':
    test_dataset()