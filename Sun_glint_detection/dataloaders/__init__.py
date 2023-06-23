import imp
import os
from dataloaders.dataloader import make_dataset
from torch.utils.data import DataLoader
from utils.mypath import Path

def make_data_loader(args):

    if args.dataset == 'sun_glint':
        train_set = make_dataset(txt=os.path.join(Path.db_root_dir(args.dataset),'train.txt'))
        val_set = make_dataset(txt=os.path.join(Path.db_root_dir(args.dataset),'val.txt'))
        test_set = make_dataset(txt=os.path.join(Path.db_root_dir(args.dataset),'test.txt'))
        train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


    else:
        raise NotImplementedError