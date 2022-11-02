from code.efanet import EFANet
from code.autoint import AutoInt
from code.fignn import FiGNN
from code.fint import FINT
from code.InterHAt import InterHAt
from code.xcrossnet import XCrossNet
from code.frnet import FRNet
from code.adnfm import AdnFM
import numpy as np
from time import time
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def str2list(v):
    v=v.split(',')
    v=[int(_.strip('[]')) for _ in v]

    return v


def str2list2(v):
    v=v.split(',')
    v=[float(_.strip('[]')) for _ in v]

    return v


def str2bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='GraphFM')
    parser.add_argument('--is_save', action='store_true', default=True)
    parser.add_argument('--greater_is_better', action='store_true', help='early stop criterion')
    parser.add_argument('--has_residual', action='store_true', help='add residual')
    parser.add_argument('--weight_limits', type=str2list, default=[0.0256, 0.0526, 0.1111], help='2131241')
    parser.add_argument('--k', type=int, default=15, help='keep the top k nodes')
    parser.add_argument('--blocks', type=int, default=3, help='#blocks')
    parser.add_argument('--block_shape', type=str2list, default=[64, 64, 64], help='output shape of each block')
    parser.add_argument('--ks', type=str2list, default=[39, 20, 5], help='the size of sampled neighborhood')
    parser.add_argument('--heads', type=int, default=2, help='#heads')
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--dropout_keep_prob', type=str2list2, default=[1, 1, 0.5])
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--l2_reg', type=float, default=0.0)
    parser.add_argument('--random_seed', type=int, default=2018)
    parser.add_argument('--save_path', type=str, default='./model/')
    parser.add_argument('--field_size', type=int, default=39, help='#fields')
    parser.add_argument('--loss_type', type=str, default='logloss')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--run_times', type=int, default=3,help='run multiple times to eliminate error')
    parser.add_argument('--deep_layers', type=str2list, default=[64, 32, 10], help='config for dnn in joint train')
    parser.add_argument('--batch_norm', type=int, default=0)
    parser.add_argument('--batch_norm_decay', type=float, default=0.995)
    parser.add_argument('--data', type=str, help='data name', default="criteo")
    parser.add_argument('--data_path', type=str, help='root path for all the data', default="E:\\dataset")
    return parser.parse_args()


def _run_(args, file_name):
    path_prefix = os.path.join(args.data_path, args.data)
    feature_size = np.load(path_prefix + '/feature_size.npy')[0]

    if args.model_type == 'GraphFM':
        model = GraphFM(args=args, feature_size=feature_size)
    elif args.model_type == 'AutoInt':
        model = AutoInt(args=args, feature_size=feature_size)
    elif args.model_type == 'InterHAt':
        model = InterHAt(args=args, feature_size=feature_size)
    elif args.model_type == 'FiGNN':
        model = FiGNN(args=args, feature_size=feature_size)
    elif args.model_type == 'FINT':
        model = FINT(args=args, feature_size=feature_size)
    elif args.model_type == 'EFANet':
        model = EFANet(args=args, feature_size=feature_size)
    elif args.model_type == 'XCrossNet':
        model = XCrossNet(args=args, feature_size=feature_size)
    elif args.model_type == 'FRNet':
        model = FRNet(args=args, feature_size=feature_size)
    elif args.model_type == 'AdnFM':
        model = AdnFM(args=args, feature_size=feature_size)

    print('start testing!...')
    Xi_test = np.load(path_prefix + '/part1/' + file_name[0])
    Xv_test = np.load(path_prefix + '/part1/' + file_name[1])
    y_test = np.load(path_prefix + '/part1/' + file_name[2])

    if args.is_save == True: model.restore()

    test_result, test_loss = model.evaluate(Xi_test, Xv_test, y_test)
    print("test-result = %.4lf, test-logloss = %.4lf" % (test_result, test_loss))

    return test_result, test_loss


if __name__ == "__main__":
    args = parse_args()
    print(args.__dict__)
    print('**************')
    if args.data in ['Avazu', 'Wiki']:
        # Avazu does not have numerical features so we didn't scale the data.
        file_name = ['train_i.npy', 'train_x.npy', 'train_y.npy']
    elif args.data in ['criteo', 'KDD2012']:
        file_name = ['train_i.npy', 'train_x2.npy', 'train_y.npy']
    test_auc = []
    test_log = []

    print('run time : %d' % args.run_times)
    test_result, test_loss = _run_(args, file_name)
    test_auc.append(test_result)
    test_log.append(test_loss)
    print('test_auc', test_auc)
    print('test_log_loss', test_log)
    print('avg_auc', sum(test_auc)/len(test_auc))
    print('avg_log_loss', sum(test_log)/len(test_log))

