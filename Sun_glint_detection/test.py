import argparse
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from utils.mypath import Path
from dataloaders import make_data_loader
from models.sync_batchnorm.replicate import patch_replication_callback
from models import build_model
from utils.metrics import Evaluator

def draw_pred(label_pred, num_classes,palette):
        c,h,w=label_pred.shape
        label=np.zeros((h,w,3))
        for j in range(num_classes):
            label[label_pred[0]==j]=palette[j]
        return label

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.nclass=2
        self.palette=[[0,0,0],[255,255,255]]

        if args.cuda:
            self.device='cuda'
        else:
            self.device='cpu'

        self.dtype=torch.float32

        # Define Dataloader
        self.train_loader, self.val_loader, self.test_loader= make_data_loader(args)

        # Define network
        model = build_model(args)

        log_dir = os.path.join('./run', args.dataset, args.checkname, 'model_best.pth.tar')
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['state_dict'])

        self.model = model
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
        patch_replication_callback(self.model)
        self.model=self.model.to(device=self.device)

        if args.ft:
            args.start_epoch = 0


    def test(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        for i,(fn,slice_img,slice_label) in enumerate(tbar):
            slice_img, slice_label = slice_img.to(device=self.device,dtype=self.dtype), slice_label.to(device=self.device)
            scores=self.model(slice_img)

            N,_,h,w=scores.shape
            label_pred=scores.max(dim=1)[1].cpu().numpy()
            label_true=slice_label.cpu().numpy()

            pred=draw_pred(label_pred,self.nclass,self.palette)
            pred = pred.astype(np.uint8)
            basename=os.path.basename(str(fn))
            save_path='./run/sun_light/fassnet/experiment_0/pred'
            name=str(save_path)+'/'+basename[:-3]
            cv2.imwrite(name,pred)


            self.evaluator.add_batch(label_true,label_pred)

        test_accuracy=self.evaluator.Pixel_Accuracy_Mean()
        test_class_acc=self.evaluator.Pixel_Accuracy_Class()
        test_class_iou,test_miou=self.evaluator.Mean_Intersection_over_Union()

        test_result=str('Epoch:{} Test_acc:{} Test_iou:{}'\
            .format(epoch
                    ,test_class_acc
                    ,test_class_iou))

        str_test='Test_acc:'+str(np.round(test_accuracy,5))+' '\
                  +'Test_miou:'+str(np.round(test_miou,5))+' '
        print(test_result)
        print(str_test)

def main():
    parser = argparse.ArgumentParser(description="Sun Glint Testing")
    parser.add_argument('--model', type=str, default='fassnet',
                        choices=['fassnet'],
                        help='model name (default: fassnet)')
    parser.add_argument('--dataset', type=str, default='sun_glint',
                        choices=['sun_glint', 'sun_glint_aug'],
                        help='dataset name')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--val_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                val (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'sun_glint': 1        
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.train_batch_size is None:
        args.train_batch_size = 8 * len(args.gpu_ids)

    if args.val_batch_size is None:
        args.val_batch_size = args.train_batch_size/4

    if args.test_batch_size is None:
        args.test_batch_size = args.gpu_ids


    if args.checkname is None:
        args.checkname = str(args.model)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.test(epoch)

if __name__ == "__main__":
   main()


