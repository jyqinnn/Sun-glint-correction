import os
from tqdm import tqdm
import numpy as np
from utils.mypath import Path
from dataloaders.dataloader import make_dataset
from torch.utils.data import DataLoader
from tempfile import TemporaryFile

def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for slice_img,slice_label in tqdm_batch:
        y = slice_label.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset + '_classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret

if __name__ == "__main__":
    train_data = make_dataset(txt='./train.txt', img_transform=None
                              )
    validation_data = make_dataset(txt='./val.txt', img_transform=None
                                   )
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=validation_data, batch_size=2, shuffle=True)
    weight=calculate_weigths_labels('sun_glint',train_loader,2)
    print(weight)