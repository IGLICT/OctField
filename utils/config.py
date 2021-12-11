
import argparse



def add_base_args(parser):
    parser.add_argument('--exp_name', type=str, default='no_name', help='name of the training run')
    parser.add_argument('--category', type=str, default='Chair', help='object category')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility)')
    return parser

def add_model_args(parser):
    parser.add_argument('--model_path', type=str, default='../data/models')

    return parser


def add_data_args(parser):
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--check_valid', action='store_true', default=False)
    parser.add_argument('--num_smp', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--res', type=int, default=4096)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--vres', type=int, default=16)
    parser.add_argument('--sdf', action='store_true', default=False)
    parser.add_argument('--hdf5', action='store_true', default=False)
    parser.add_argument('--load_ram', action='store_true', default=False)

    return parser


def add_train_vae_args(parser):
    parser = add_base_args(parser)
    parser = add_model_args(parser)
    parser = add_data_args(parser)

    # validation dataset
    parser.add_argument('--train_list', type=str, default='train.txt', help='file name for the list of object names for training')
    parser.add_argument('--train_dataset', type=str, default='train.h5', help='hdf5 file of objects for training')
    parser.add_argument('--val_list', type=str, default='val.txt', help='file name for the list of object names for validation')
    parser.add_argument('--val_dataset', type=str, default='val.h5', help='hdf5 file of objects for validation')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--lr_decay_every', type=int, default=500)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--cut', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--min_value', type=float, default=0.)
    parser.add_argument('--max_value', type=float, default=0.99)
    
    parser.add_argument('--non_variational', action='store_true', default=False, help='make the variational autoencoder non-variational')

    parser.add_argument('--loss_weight_center', type=float, default=0.0) #20
    parser.add_argument('--loss_weight_leaf', type=float, default=0.1)
    parser.add_argument('--loss_weight_latent', type=float, default=10.0)#10
    parser.add_argument('--loss_weight_scale', type=float, default=0.0)#20
    parser.add_argument('--loss_weight_type', type=float, default=0.1)
    parser.add_argument('--loss_weight_geo', type=float, default=1.0)
    parser.add_argument('--loss_weight_kldiv', type=float, default=0.02) #0.05


    parser.add_argument('--mc_res', type=int, default=32)

    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--geo_feat_size', type=int, default=32)
    parser.add_argument('--child_feats_size', type=int, default=256)

    parser.add_argument('--children_hidden_size', type=int, default=256)
    parser.add_argument('--geo_hidden_size', type=int, default=256)
    parser.add_argument('--sample_hidden_size', type=int, default=256)
    parser.add_argument('--shape_hidden_size', type=int, default=256)
    parser.add_argument('--type_hidden_size', type=int, default=256)

    # logging
    parser.add_argument('--log_path', type=str, default='../data/logs')
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=3, help='number of optimization steps beween console log prints')
    parser.add_argument('--checkpoint_interval', type=int, default=500, help='number of optimization steps beween checkpoints')

    parser.add_argument('--model_version', type=str, default='model', help='model file name')
    # load pretrained model (for pc exps)
    parser.add_argument('--part_pc_exp_name', type=str, help='resume model exp name')
    parser.add_argument('--part_pc_model_epoch', type=int, help='resume model epoch')
    return parser


def add_result_args(parser):
    parser = add_base_args(parser)
    parser = add_model_args(parser)

    parser.add_argument('--result_path', type=str, default='../data/results')
    parser.add_argument('--model_epoch', type=int, default=-1, help='model at what epoch to use (set to < 0 for the final/most recent model)')

    return parser


def add_eval_args(parser):
    parser = add_result_args(parser)
    parser = add_data_args(parser)
    parser.add_argument('--test_list', type=str, default='val.txt', help='file name for the list of object names for testing')
    parser.add_argument('--test_dataset', type=str, default='val.h5', help='hdf5 file of objects for testing')

    return  parser


def add_sample_args(parser):
    parser.add_argument('--num_smp', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--res', type=int, default=4096)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--data_source', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_list', type=str, required=True)
    parser.add_argument('--success_list', type=str, default='success.txt')
    parser.add_argument('--fail_list', type=str)
    return parser




def add_check_args(parser):
    parser.add_argument('--num_smp', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--res', type=int, default=4096)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--data_source', type=str, required=True)
    parser.add_argument('--success_list', type=str, default='success.txt')
    return parser



def add_voxelize_args(parser):
    parser.add_argument('--num_smp', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--res', type=int, default=4096)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--vres', type=int, default=16)
    parser.add_argument('--data_source', type=str, required=True)
    parser.add_argument('--success_list', type=str, default='success.txt')
    return parser


def add_prepare_args(parser):
    parser.add_argument('--num_smp', type=int, default=10000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--res', type=int, default=50000)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--vres', type=int, default=32)
    parser.add_argument('--num_pc', type=int, default=2048)
    parser.add_argument('--data_source', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_list', type=str, required=True)
    parser.add_argument('--log_path', type=str, default='.')
    parser.add_argument('--success_list', type=str, default='success.txt')
    parser.add_argument('--fail_list', type=str, default='fail.txt')
    return parser
