import numpy as np


def create_z_score_norm(dataset):
    return np.mean(dataset, axis=0), np.std(dataset, axis=0)


def apply_z_score_norm(dataset, mean, std):
    np.divide(np.subtract(dataset, mean), std)


def z_norm(ref_dataset, datasets):
    mean, std = create_z_score_norm(ref_dataset)
    for i, dataset in enumerate(datasets):
        apply_z_score_norm(datasets[i], mean, std)
