
import argparse


def parse_args_eposide_train():
    parser = argparse.ArgumentParser(description='eposide_train')
    parser.add_argument('--source_data_path', default='/data1/zhoufei/CDFSL/datasets/miniImagenet/train', help='train data path')
    parser.add_argument('--image_size'  , default=224, type=int,  help='image size')
    parser.add_argument('--train_num_class' , default=64, type=int, help='total number of classes in in train class')
    parser.add_argument('--list_of_out_dims', default=[64,128,256,512], help='every block output')
    parser.add_argument('--list_of_stride', default=[1,2,2,2], help='every block conv stride')
    parser.add_argument('--list_of_dilated_rate', default=[1,1,1,1], help='dilated conv') 
    parser.add_argument('--train_aug', default='True',  help='perform data augmentation or not during training ') 
    parser.add_argument('--save_dir', default='./logs', help='Save dir')
    parser.add_argument('--epoch', default=100, type=int, help ='total batch train epoch')  
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--n_way', default=5, type=int,  help='class num to classify for every task')
    parser.add_argument('--n_support', default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--n_query', default=15, type=int,  help='number of test data in each class, same as n_query') 
    parser.add_argument('--train_n_eposide', default=100, type=int, help ='total task every epoch') 
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--m', type=float, default=0.998, help='epsilon of moment')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--lamba2', type=float, default=0.15, help='lamba_intra')
    parser.add_argument('--lamba1', type=float, default=1.0, help='lamba_cross') 
    parser.add_argument('--pretrain_model_path', default='./pretrain/399.tar', help='model_path')

    return parser.parse_args()




def parse_args_test():
    
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--image_size'  , default=224, type=int,  help='image size') 
    parser.add_argument('--feature_size' , default=512, type=int, help='feature_size')
    parser.add_argument('--list_of_out_dims', default=[64,128,256,512], help='every block output')
    parser.add_argument('--list_of_stride', default=[1,2,2,2], help='every block conv stride')
    parser.add_argument('--list_of_dilated_rate', default=[1,1,1,1], help='dilated conv') 
    parser.add_argument('--n_way', default=5, type=int,  help='class num to classify for every task')
    parser.add_argument('--n_support', default=5, type=int,  help='number of labeled data in each class, same as n_support') 
    parser.add_argument('--n_query', default=15, type=int,  help='number of test data in each class, same as n_query') 
    parser.add_argument('--test_n_eposide', default=600, type=int, help ='total task every epoch') 
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--current_data_path', default='/data0/datasets/ISIC', help='ISIC_data_path')
    parser.add_argument('--current_class', default=7, type=int, help='total number of classes in ISIC')
    parser.add_argument('--pretrain_model_path', default='./pretrain/399.tar', help='pretrain_model_path')
    parser.add_argument('--model_path', default='./checkpoint/100.tar', help='model_path')
    
    return parser.parse_args()










