import argparse
import os
import pickle
import json
from datautils import load_EEG,load_EEG_with_img_name
from transformers import CLIPModel, CLIPProcessor

# MODEL_PATH = "clip-vit-large-patch14"
# clip_model = CLIPModel.from_pretrained(MODEL_PATH)  # 全局模型
# clip_processor = CLIPProcessor.from_pretrained(MODEL_PATH)  # 全局处理器

parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--save_path', type=str, default='exp/epilepsy/test')
parser.add_argument('--dataset', type=str, default='eeg')
parser.add_argument('--data_path', type=str,
                    default='data/EEG/')
#parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=128)

# --- 新增 time_low 和 time_high 参数定义 ---
parser.add_argument('-tl', '--time_low', type=int, default=20,
                    help="截取 EEG 信号的起始时间点 (例如，对应 train_freqencoder.py 中的设置)")
parser.add_argument('-th', '--time_high', type=int, default=460,
                    help="截取 EEG 信号的结束时间点 (例如，对应 train_freqencoder.py 中的设置)")
# --- 新增结束 ---

# --- 新增 model_type 参数定义 ---
parser.add_argument('-mt', '--model_type', type=str, default='eegnet', # 设置默认值为 'eegnet'
                    choices=['lstm', 'eegnet', 'FreqEncoder', 'EEGNetEncoder'], # 允许的选项
                    help='指定使用的频率模型: lstm/FreqEncoder | eegnet/EEGNetEncoder')
# --- 新增结束 ---

# --- 新增 model_params 参数定义 (如果 process.py 中确实需要它) ---
parser.add_argument('-mp','--model_params', default=[], nargs='*', # default=[] 避免 None
                    help='(可选) 频率模型特定参数 key=value 列表 (例如: F1=8 D=2)')
# --- 新增结束 ---

# --- 新增 pretrained_net 参数定义 (如果 process.py 中确实需要它) ---
parser.add_argument('--pretrained_net', default='', type=str,
                    help="预训练频率模型权重路径")
# --- 新增结束 ---

# model args
parser.add_argument('--d_model', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--attn_heads', type=int, default=16)
parser.add_argument('--eval_per_steps', type=int, default=16)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--layers', type=int, default=8)
parser.add_argument('--alpha', type=float, default=4.0)
parser.add_argument('--beta', type=float, default=2.0)

parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--vocab_size', type=int,  default=660)
parser.add_argument('--wave_length', type=int, default=4)
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--reg_layers', type=int, default=4)

# train args
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_decay_rate', type=float, default=1.)
parser.add_argument('--lr_decay_steps', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_epoch_pretrain', type=int, default=500)
parser.add_argument('--num_epoch', type=int, default=400)
parser.add_argument('--load_pretrained_model', type=int, default=1)

parser.add_argument('--fusion_dim', type=int, default=512, help='特征融合后的维度')
parser.add_argument('--learnable_alpha', action='store_true', default=False, help='是否学习融合权重 alpha (默认不学习，使用固定值)')
parser.add_argument('--fixed_alpha', type=float, default=0.5, help='如果 learnable_alpha 为 False，使用的固定 alpha 值')
parser.add_argument('--fusion_dropout', type=float, default=0.5, help='融合层后的 Dropout 比率')

parser.add_argument('--mlp_hidden_dims', nargs='+', type=int, default=[1024, 512, 256], help='MLP 隐藏层维度列表')
parser.add_argument('--mlp_dropout', type=float, default=0.5, help='MLP Dropout 比率')
parser.add_argument('--mlp_lr', type=float, default=1e-4, help='MLP 学习率')
parser.add_argument('--mlp_epochs', type=int, default=100, help='MLP 训练轮数')
parser.add_argument('--mlp_batch_size', type=int, default=256, help='MLP 训练批次大小')
parser.add_argument('--mlp_patience', type=int, default=15, help='MLP 早停耐心')

# === Ablation parameters ===
# Category C: AlignNet parameters
parser.add_argument('--alignnet_num_blocks', type=int, default=3,
                    help='C1 ablation: number of residual blocks in AlignNet (0-4)')
parser.add_argument('--alignnet_expansion', type=int, default=4,
                    help='C2 ablation: hidden expansion factor in AlignNet')
parser.add_argument('--alignnet_activation', type=str, default='tanh',
                    choices=['tanh', 'relu', 'gelu', 'leakyrelu'],
                    help='C3 ablation: activation function in AlignNet')
# Category D: Component ablations
parser.add_argument('--loss_type', type=str, default='negclip',
                    choices=['clip', 'negclip', 'tripletclip'],
                    help='D1 ablation: contrastive loss type for CLIP alignment')
parser.add_argument('--use_classifier_logits', type=int, default=1,
                    help='D5a ablation: whether to concat classifier logits in AlignNet (1=True, 0=False)')

# Train stage selection (for ablation automation)
parser.add_argument('--train_stage', type=str, default=None,
                    choices=['pretrain', 'finetune', 'finetune_timefreq', 'finetune_CLIP', 'test'],
                    help='Which training stage to run (overrides main.py defaults)')

args, _ = parser.parse_known_args()

# # 将 CLIP 模型和处理器添加到 args 中，方便其他模块调用
# args.clip_model = clip_model
# args.clip_processor = clip_processor

if args.data_path is None:
    if args.dataset == 'eeg':
        Train_data_all, Train_data, Test_data = load_EEG()
        ## Load from file
        # with open("data/EEG_divided/Train_data_all_subj2.pkl", "rb") as f:
        #     Train_data_all = pickle.load(f)
        # with open("data/EEG_divided/Train_data_subj2.pkl", "rb") as j:
        #     Train_data = pickle.load(j)
        # with open("data/EEG_divided/Test_data_subj2.pkl", "rb") as k:
        #     Test_data = pickle.load(k)
        args.num_class = len(set(Train_data[1]))
else:
    if args.dataset == 'eeg':
        path = args.data_path
        Train_data_all, Train_data, Test_data = load_EEG()
        ## Load from file
        # with open("data/EEG_divided/Train_data_all_subj2.pkl", "rb") as f:
        #     Train_data_all = pickle.load(f)
        # with open("data/EEG_divided/Train_data_subj2.pkl", "rb") as j:
        #     Train_data = pickle.load(j)
        # with open("data/EEG_divided/Test_data_subj2.pkl", "rb") as k:
        #      Test_data = pickle.load(k)
        Train_data_all_with_image_name, Train_data_with_image_name, Test_data_with_image_name = load_EEG_with_img_name()
        args.num_class = len(set(Train_data[1]))

args.eval_per_steps = max(1, int(len(Train_data[0]) / args.train_batch_size))
args.lr_decay_steps = args.eval_per_steps
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()
