import os
from tqdm import tqdm
import numpy as np
from mypath import Path

def calculate_weigths_labels(dataset, dataloader, num_classes, args = None, classes_weights_path = None):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    cnt = 1
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
        cnt += 1
        if cnt > 200:
            break
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    if classes_weights_path is None:
        if not args.dataset_dir is None:
            classes_weights_path = os.path.join(args.dataset_dir, dataset+f'_c{num_classes}_weights.npy')
        else:
            classes_weights_path = os.path.join(Path.db_root_dir(dataset, args), dataset+f'_c{num_classes}_weights.npy')
    # if args.master:
    #     np.save(classes_weights_path, ret)

    return ret

    