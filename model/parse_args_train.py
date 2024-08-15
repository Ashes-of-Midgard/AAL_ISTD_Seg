from model.utils import *


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')

    parser.add_argument('--save_inter', action='store_true')
    parser.add_argument('--loss', type=str, default='SoftIoULoss')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--test_mode', type=str, default='clean', choices=['clean', 'FGSM', 'random_noise'])
    parser.add_argument('--ema', action='store_true')

    # choose model
    parser.add_argument('--model', type=str, default='UNet')
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--attention', type=str, default='Res_block')
    parser.add_argument('--downlayer', type=str, default='two',
                        help='three, four')
    parser.add_argument('--supervision', type=str, default='Full')
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--iff_back_num', type=int, default=1)


    # parameter for DNANet
    parser.add_argument('--channel_size', type=str, default='two',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')
    parser.add_argument('--deep_supervision', type=str, default='False', help='True or False (model==DNANet)')



    # data and pre-process
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--lr_mode', type=str, default='fixed_lr', help='lr_mode name:  adjusted_lr, fixed_lr')
    parser.add_argument('--test_size', type=float, default='0.5', help='when mode==Ratio')
    parser.add_argument('--root', type=str, default='./dataset')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='default_split')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='crop image size')


    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        evaluation (default: 32)')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adagrad',
                        help=' Adam, Adagrad, SGD')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['Cosin eAnnealingLR', 'ReduceLROnPlateau', 'StepLR'])
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.1)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    args = parser.parse_args()

    if not args.only_test:
        # make dir for save result
        args.save_dir = make_dir(args.gpus, args.deep_supervision, args.dataset, args.model)

        # save training log
        save_train_log(args, args.save_dir)

    else:
        args.save_dir = None

    # the parser
    return args
