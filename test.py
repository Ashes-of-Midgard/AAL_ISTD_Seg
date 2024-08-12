# torch and visulization
import os
import time

from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader, random_split
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils  import *
from model.metric import *
from model.loss   import *
from model.load_param_data import load_dataset, load_param

# model

from model.net import (LightWeightNetwork, LightWeightNetwork_AAL, LightWeightNetwork_FGSM,
                       LightWeightNetwork_FGSM_SA, LightWeightNetwork_RA, LightWeightNetwork_SA)

import scipy.io as scio


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, 10, args.crop_size)
        self.mIoU = mIoU(1)

        # Read image index from TXT
        if args.mode == 'TXT':
            self.train_dataset_dir = args.root + '/' + args.dataset
            self.test_dataset_dir  = args.root + '/' + args.dataset

        self.train_img_ids, self.val_img_ids, self.test_img_ids = load_dataset(args.root, args.dataset, args.split_method)
        #self.train_img_ids, self.val_img_ids, self.test_txt = load_dataset(args.root, args.dataset, args.split_method)

        if args.dataset=='ICPR_Track2':
            mean_value = [0.2518, 0.2518, 0.2519]
            std_value  = [0.2557, 0.2557, 0.2558]
        elif args.dataset=='IRSTD-1k':
            mean_value = [.485, .456, .406]
            std_value = [.229, .224, .225]
        elif args.dataset=='NUDT-SIRST':
            mean_value = [.485, .456, .406]
            std_value = [.229, .224, .225]
        elif args.dataset=='NUAA-SIRST-v2':
            mean_value = [111.89, 111.89, 111.89]
            std_value = [27.62, 27.62, 27.62]

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_value, std_value)])
        
        evalset = TestSetLoader(self.test_dataset_dir, img_id=self.val_img_ids,  base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)

        self.eval_data  = DataLoader(dataset=evalset,  batch_size=args.eval_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model == 'UNet':
            model = LightWeightNetwork()
        elif args.model == 'UNet-AAL':
            model = LightWeightNetwork_AAL()
        elif args.model == 'UNet-FGSM':
            model = LightWeightNetwork_FGSM()
        elif args.model == 'UNet-FGSM-SA':
            model = LightWeightNetwork_FGSM_SA()
        elif args.model == 'UNet-SA':
            model = LightWeightNetwork_SA()
        elif args.model == 'UNet-RA':
            model = LightWeightNetwork_RA()

        # Load trained model
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'])

        model = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model = model

    def evaluation(self,epoch):
        # Load trained model
        # Test
        self.model.eval()
        tbar = tqdm(self.eval_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, (data, labels, img_sizes) in enumerate(tbar):
                data   = data.cuda()
                labels = labels.cuda()
                pred = self.model(data)
                loss = SoftIoULoss(pred, labels)
                #save_Pred_GT_for_split_evalution(pred, labels, target_image_path, self.val_img_ids, num, args.suffix, args.crop_size)
                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)
                _, mean_IOU = self.mIoU.get()

            FA, PD    = self.PD_FA.get(len(self.val_img_ids), args.crop_size)
            test_loss = losses.avg

            print('test_loss, %.4f' % (test_loss))
            print('mean_IOU:', mean_IOU)
            print('PD:', PD)
            print('FA:', FA)
            self.best_iou = mean_IOU

def main(args):
    trainer = Trainer(args)
    trainer.evaluation(epoch=None)

if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)





