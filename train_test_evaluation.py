# torch and visulization
import os
import time

from tqdm             import tqdm
from torch import Tensor
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader, random_split
from model.parse_args_train import  parse_args
from typing import Tuple, List
import matplotlib

# metric, loss .etc
from model.utils  import *
from model.metric import *
from model.loss   import *
from model.load_param_data         import  load_dataset, load_param

# model

from model.net import (LightWeightNetwork, LightWeightNetwork_AAL, LightWeightNetwork_FGSM,
                       LightWeightNetwork_FGSM_SA, LightWeightNetwork_RN, LightWeightNetwork_SA,
                       LightWeightNetwork_IFF)

import scipy.io as scio


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, 10, args.crop_size)
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir = args.save_dir
        nb_filter, num_blocks= load_param(args.channel_size, args.backbone)

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
        else:
            mean_value = [0,0,0]
            std_value=[1,1,1]
        self.mean_value = mean_value
        self.std_value = std_value

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_value, std_value)])
        
        trainset = TestSetLoader(self.train_dataset_dir, img_id=self.train_img_ids, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        testset = TestSetLoader(self.train_dataset_dir, img_id=self.val_img_ids, base_size=args.base_size, crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        evalset = TestSetLoader(self.test_dataset_dir, img_id=self.val_img_ids,  base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)

        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)
        self.eval_data  = DataLoader(dataset=evalset,  batch_size=args.eval_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model == 'UNet':
            model = LightWeightNetwork()
        elif args.model == 'UNet-AAL':
            model = LightWeightNetwork_AAL(eps=args.eps)
        elif args.model == 'UNet-FGSM':
            model = LightWeightNetwork_FGSM(eps=args.eps)
        elif args.model == 'UNet-FGSM-SA':
            model = LightWeightNetwork_FGSM_SA(eps=args.eps)
        elif args.model == 'UNet-SA':
            model = LightWeightNetwork_SA(eps=args.eps)
        elif args.model == 'UNet-RN':
            model = LightWeightNetwork_RN(eps=args.eps)
        elif args.model == 'UNet-IFF':
            model = LightWeightNetwork_IFF(iff_back_num=args.iff_back_num)

        if args.model_dir is not None:
            checkpoint = torch.load(args.model_dir)
            model.load_state_dict(checkpoint['state_dict'])
            model = model.cuda()
            print(f'Model loaded from {args.model_dir}')
        else:
            model = model.cuda()
            model.apply(weights_init_xavier)
            print("Model Initializing")
        self.model = model

        # Optimizer and lr scheduling

        if args.optimizer == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        if args.scheduler == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)


        # DATA_Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Loss
        if args.loss == 'SoftIoULoss':
            self.loss_fn = SoftIoULoss
        elif args.loss == 'SLSIoULoss':
            self.loss_fn = SLSIoULoss(int(0.25*args.epochs))

    # Training
    def training(self, epoch):
        lr = self.scheduler.get_lr()[0]
        save_lr_dir = 'result_WS/' + self.save_dir + '/' + self.save_prefix + '_learning_rate.log'
        with open(save_lr_dir, 'a') as f:
            f.write(' learning_rate: {:04f}:\n'.format(lr))
        print('learning_rate:',lr)


        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, ( data, labels, img_sizes) in enumerate(tbar):
            data   = data.cuda()
            labels = labels.cuda()
            if type(self.model) in (LightWeightNetwork_AAL, LightWeightNetwork_FGSM, LightWeightNetwork_FGSM_SA):
                if type(self.loss_fn) == SLSIoULoss:
                    pred = self.model(data, labels, lambda x,y:self.loss_fn(x,y,epoch=epoch))
                else:
                    pred = self.model(data, labels, self.loss_fn)
                if i==0 and args.save_inter and type(self.model) == LightWeightNetwork_AAL:
                    os.makedirs('./result_WS/'+args.save_dir+'/'+'inter_results',exist_ok=True)
                    size = [img_sizes[0][0].item(), img_sizes[0][1].item()]
                    
                    ori_img = tensor_to_img(de_normalize(data[0], self.mean_value, self.std_value),size)
                    mask = tensor_to_img(labels[0], size)
                    pred_mask = tensor_to_img(torch.sigmoid(pred[0]), size)

                    sa = sa_heat_map(self.model.reserved_sa[0]).resize(size)
                    back_mask = sa_heat_map(self.model.reserved_back_mask[0]).resize(size)
                    backtracked_sa = sa_heat_map(self.model.reserved_backtracked_sa[0]).resize(size)
                    sa_overlayed = sa_over_img(sa,
                                               ori_img)
                    backtracked_sa_overlayed = sa_over_img(backtracked_sa,
                                                           ori_img)
                    
                    delta = tensor_to_img(self.model.reserved_delta[0],size)
                    attacked_img = tensor_to_img(de_normalize(self.model.reserved_attacked_img[0],self.mean_value,self.std_value),size)
                    attacked_img_sa = tensor_to_img(de_normalize(self.model.reserved_attacked_img_sa[0],self.mean_value,self.std_value),size)
                    attacked_img_backtracked = tensor_to_img(de_normalize(self.model.reserved_attacked_img_backtracked_sa[0],self.mean_value,self.std_value),size)
                    
                    ori_img.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'ori_img_'+str(epoch)+'.png')
                    mask.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'mask_'+str(epoch)+'.png')
                    pred_mask.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'pred_'+str(epoch)+'.png')

                    sa.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'sa_'+str(epoch)+'.png')
                    back_mask.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'back_mask_'+str(epoch)+'.png')
                    backtracked_sa.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'backtracked_sa_'+str(epoch)+'.png')
                    sa_overlayed.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'sa_over_'+str(epoch)+'.png')
                    backtracked_sa_overlayed.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'backtracked_sa_over_'+str(epoch)+'.png')
                    
                    delta.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'delta_'+str(epoch)+'.png')
                    attacked_img.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'img_attacked_'+str(epoch)+'.png')
                    attacked_img_sa.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'img_attacked_sa_'+str(epoch)+'.png')
                    attacked_img_backtracked.save('./result_WS/'+args.save_dir+'/'+'inter_results'+'/'+'img_attacked_backtracked_'+str(epoch)+'.png')        
            else:
                pred = self.model(data)
            if type(self.loss_fn)==SLSIoULoss:
                loss = self.loss_fn(pred, labels, epoch)
            else:
                loss = self.loss_fn(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.train_loss = losses.avg
        if args.lr_mode == 'adjusted_lr':
            self.scheduler.step()


    # Testing
    def testing (self, epoch):
        tbar   = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, (data, labels, img_sizes) in enumerate(tbar):
                data   = data.cuda()
                labels = labels.cuda()
                pred   = self.model(data)
                if type(self.loss_fn)==SLSIoULoss:
                    loss = self.loss_fn(pred, labels, epoch)
                else:
                    loss = self.loss_fn(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC.update(pred, labels)
                self.mIoU.update(pred, labels)
                _, mean_IOU = self.mIoU.get()
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU))

            self.test_loss = losses.avg

            save_train_test_loss_dir = 'result_WS/' + self.save_dir + '/' + self.save_prefix + '_train_test_loss.log'
            with open(save_train_test_loss_dir, 'a') as f:
                f.write('epoch: {:04f}:\t'.format(epoch))
                f.write('train_loss: {:04f}:\t'.format(self.train_loss))
                f.write('test_loss: {:04f}:\t'.format(self.test_loss))
                f.write('\n')

        # save high-performance model
        if mean_IOU > self.best_iou:
            self.best_iou = mean_IOU
            save_model(self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, self.test_loss, recall, precision, epoch, self.model.state_dict())

    def evaluation(self,epoch):
        candidate_model_dir = os.listdir('result_WS/' + self.save_dir )
        for model_num in range(len(candidate_model_dir)):
            model_dir = 'result_WS/' + self.save_dir + '/' + candidate_model_dir[model_num]
            if '.pth.tar' in model_dir:
                model_path = model_dir

        checkpoint        = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to('cuda')

        evaluation_save_path =  './result_WS/' + self.save_dir
        target_image_path    =  evaluation_save_path + '/' +'visulization_result'
        target_dir           =  evaluation_save_path + '/' +'visulization_fuse'

        make_visulization_dir(target_image_path, target_dir)

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
                if type(self.loss_fn)==SLSIoULoss:
                    loss = self.loss_fn(pred, labels, epoch)
                else:
                    loss = self.loss_fn(pred, labels)
                #save_Pred_GT_for_split_evalution(pred, labels, target_image_path, self.val_img_ids, num, args.suffix, args.crop_size)
                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)
                _, mean_IOU = self.mIoU.get()

            FA, PD    = self.PD_FA.get(len(self.val_img_ids), args.crop_size)
            test_loss = losses.avg
            scio.savemat(evaluation_save_path + '/' + 'PD_FA_' + str(255), {'number_record1': FA, 'number_record2': PD})

            print('test_loss, %.4f' % (test_loss))
            print('mean_IOU:', mean_IOU)
            print('PD:', PD)
            print('FA:', FA)
            self.best_iou = mean_IOU


def tensor_to_img(input:Tensor, img_size:Tuple[int, int]) -> Image.Image:
    #img_array = (input * 255).type(torch.uint8)
    #img_array = img_array.permute([1,2,0])
    #img_array = np.array(img_array.detach().cpu())
    img = transforms.ToPILImage()(input)
    #if img_array.shape[2] == 3:
    #    img = Image.fromarray(img_array, mode='RGB').convert('L')
    #elif img_array.shape[2] == 1:
    #    img_array = img_array[:,:,0]
    #    img = Image.fromarray(img_array, mode='L')
    img = img.resize(img_size)
    return img


def de_normalize(input:Tensor, mean_value:List, std_value:List) -> Tensor:
    output = torch.zeros_like(input)
    output[0,:,:] = input[0,:,:] * std_value[0] + mean_value[0]
    output[1,:,:] = input[1,:,:] * std_value[1] + mean_value[1]
    output[2,:,:] = input[2,:,:] * std_value[2] + mean_value[2]
    return output


def sa_over_img(sa:Image.Image, img:Image.Image) -> Image:
    sa = sa.resize(img.size)
    overlayed_img = Image.blend(sa, img, 0.5)
    return overlayed_img


def sa_heat_map(sa:Tensor) -> Image.Image:
    assert sa.size(0) == 1
    sa = np.array(sa[0,:,:].detach().cpu())
    coolwarm_cm = matplotlib.colormaps['coolwarm']
    heat_map = coolwarm_cm(sa)
    heat_map = (heat_map[:,:,:3] * 255).astype(np.int8)
    heat_map = Image.fromarray(heat_map, mode='RGB')
    return heat_map


def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)
        if (epoch+1) ==args.epochs:
           trainer.evaluation(epoch)


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)





