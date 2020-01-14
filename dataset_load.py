import sys
sys.path.append('./datasets/c3d_dataset')
from dataset_c3d import get_training_set, get_test_set
sys.path.remove('./datasets/c3d_dataset')

sys.path.append('./datasets/lrcn_dataset')
from dataset_lrcn import get_train_data, get_test_data
sys.path.remove('./datasets/lrcn_dataset')

def generate_dataset(model_name, dataset_name):
    assert model_name in ['c3d', 'lrcn']
    if model_name == 'c3d':
        train_dataset = get_training_set(dataset_name)
        test_dataset = get_test_set(dataset_name)
    elif model_name == 'lrcn':
        train_dataset = get_train_data(dataset_name)
        test_dataset = get_test_data(dataset_name)
    return train_dataset, test_dataset
