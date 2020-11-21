import argparse


def TestOptions():
    parser = argparse.ArgumentParser(description='PyTorch hand pose Training')
    
    #dataset
    #model structure
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')
    parser.add_argument('--is-train', type=bool, default=False,
                        help='is train')
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--hpe-enabled', type=bool, default=False, help='is hpe')
    

    
    # data preprocessing
    
    
    #checkpoint
    
    return parser.parse_args()