import argparse
import torch

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str,
                        default="trial1")
    parser.add_argument("--follow_tag", type=str,
                        default="trial1")
    parser.add_argument("--data_path", type=str,
                        default="./Dataset/")
    parser.add_argument("--pre_trained_path", type=str,
                        default=None)
    parser.add_argument("--dataset", type=str, default='GTSRB', help="GTSRB; CelebA")
    parser.add_argument("--batch_size", type=int, default=2, help='2 step 1,2; 13 for step 4')
    parser.add_argument("--wm_batch_size", type=int, default=3)
    parser.add_argument("--cl_batch_size", type=int, default=2)
    parser.add_argument("--num_class", type=int, default=43, help="43 for GTSRB; 8 for CelebA")
    parser.add_argument("--output_class", type=int, default=43)
    parser.add_argument("--wm_classes", type=int, default=6, help='2 for step 3; 6 for step 1, 4')
    parser.add_argument("--wm_num", type=int, default=30)
    parser.add_argument("--cl_num", type=int, default=300)
    parser.add_argument("--stop_cl_num", type=int, default=0)
    parser.add_argument("--interval_batch", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr_optimizer_for_c", type=float, default=1e-3, help='1e-3 for Adam; 1e-2 for sgd')
    parser.add_argument("--lr_optimizer_for_t", type=float, default=1e-3, help='1e-3 for Adam; 1e-2 for sgd')
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--a", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument("--to_save", type=str, default='True')
    parser.add_argument("--to_print", type=str, default='True')
    parser.add_argument("--my_low_precision", default=torch.float16)
    parser.add_argument("--missing_proption", default=0.1)
    parser.add_argument("--log_index", default=100)
    parser.add_argument("--pretrained_name", default='vit_base_patch16_128')
    parser.add_argument("--log_name", default=[])
    parser.add_argument("--factor", default=200 / 30 * 0.45)
    parser.add_argument("--moving", default=15)
    return parser

opt = get_arguments().parse_args()
opt.proportion_cl = 0.0

opt.logpath_trigger = './log_files/log/{}{}_{}/'.format('', opt.dataset,
                                                opt.tag)

# !!! customize according to the recording results
opt.trigger_name = '20' # the name of the watermark generator's checkpoint
opt.cleanse_trigger_name = '4' #the name of the anti-watermark generator's checkpoint

#the output pathes
opt.logpath = './log_files/follow_log/{}{}_{}/'.format('(1)', opt.dataset,
                                                      opt.follow_tag)           #store the verification records
opt.logpath_data = './log_files/log_data/{}{}_{}/'.format('(1)', opt.dataset,
                                           opt.tag)                             #store the indexes of pivot samples
opt.logpath_clean = './log_files/log_clean/{}{}_{}/'.format('', opt.dataset,
                                                      opt.tag)                  #store the anti-watermark generator
opt.logpath_set_idx = './log_files/log_set_idx/{}{}_{}/'.format('(1)', opt.dataset,
                                                      opt.tag)                  #store the shuffle indexes

# set the duration and the number of each kind of watermark-related samples
opt.len_soft_wm_data = 2400
opt.soft_wm_num = 200
opt.len_hard_wm_data = 1800
opt.overall_num = 2450
opt.soft_point_num = 4
opt.hard_point_num = 4
