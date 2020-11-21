import argparse


def TrainOptions():
    parser = argparse.ArgumentParser(description='PyTorch hand pose Training')
    
    #dataset
    #model structure

    # training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='adam',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=1, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=5.0e-6, type=float,#1.0e-4(too high), 5.0e-5()
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default= 1.0e-5, type=float, #0.0005(?))
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[50,80],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')
    parser.add_argument('--is-train', type=bool, default=True,
                        help='is train')
    parser.add_argument('--gan-mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--gpu-ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--lambda-L1', type=float, default=50.0, help='weight for L1 loss')       
    



    # data preprocessing
    
    
    #checkpoint
    
    return parser.parse_args()