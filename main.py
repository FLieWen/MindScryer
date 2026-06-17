import warnings
warnings.filterwarnings('ignore')
import torch.utils.data as Data
from args import args, Test_data, Train_data_all, Train_data, Train_data_all_with_image_name, Train_data_with_image_name, Test_data_with_image_name
from dataset import Dataset,Dataset_with_image_name
from model.MindScryerModels_test import TimeEncoder,TimeFreqEncoder,FreqEncoder, EEGNetEncoder
from process import Trainer
from classification import fit_lr, get_rep_with_label,get_rep_with_label_with_image_name, MLPClassifierTorch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import argparse
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np
import os
import time # 导入 time
from datetime import datetime # 导入 datetime
from tqdm import tqdm # 导入 tqdm
import joblib # 用于保存 scaler

parser = argparse.ArgumentParser(description="Template")
parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
# Parse arguments

# --- 新增 model_type 参数定义 ---
parser.add_argument('-mt', '--model_type', type=str, default='eegnet', # 设置默认值为 'eegnet'
                    choices=['lstm', 'eegnet', 'FreqEncoder', 'EEGNetEncoder'], # 允许的选项
                    help='指定使用的频率模型: lstm/FreqEncoder | eegnet/EEGNetEncoder')
# --- 新增结束 ---

parser.add_argument('--mlp_hidden_dims', nargs='+', type=int, default=[1024, 512, 256], help='MLP 隐藏层维度列表')
parser.add_argument('--mlp_dropout', type=float, default=0.5, help='MLP Dropout 比率')
parser.add_argument('--mlp_lr', type=float, default=1e-4, help='MLP 学习率')
parser.add_argument('--mlp_epochs', type=int, default=100, help='MLP 训练轮数')
parser.add_argument('--mlp_batch_size', type=int, default=256, help='MLP 训练批次大小')
parser.add_argument('--mlp_patience', type=int, default=15, help='MLP 早停耐心')
parser.add_argument('--mlp_val_split', type=float, default=0.1, help='从训练集中划分验证集的比例')

parser.add_argument('--fusion_dim', type=int, default=512, help='特征融合后的维度')
parser.add_argument('--learnable_alpha', action='store_true', default=False, help='是否学习融合权重 alpha (默认不学习，使用固定值)')
parser.add_argument('--fixed_alpha', type=float, default=0.5, help='如果 learnable_alpha 为 False, 使用的固定 alpha 值')
parser.add_argument('--fusion_dropout', type=float, default=0.5, help='融合层后的 Dropout 比率')



opt, _ = parser.parse_known_args()

def main():
    ## Save data to local path
    ## Reduce the data load time on server for other training steps
    # with open("data/EEG_divided/Train_data_all.pkl", "wb") as f:
    #     pickle.dump(Train_data_all,f)
    #
    # with open("data/EEG_divided/Train_data.pkl", "wb") as j:
    #     pickle.dump(Train_data,j)
    #
    # with open("data/EEG_divided/Test_data.pkl", "wb") as k:
    #     pickle.dump(Test_data,k)
    torch.set_num_threads(12)
    torch.cuda.manual_seed(3407)
    train_dataset = Dataset(device=args.device, mode='pretrain', data=Train_data_all, wave_len=args.wave_length)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    args.data_shape = train_dataset.shape()
    train_linear_dataset = Dataset(device=args.device, mode='supervise_train', data=Train_data, wave_len=args.wave_length)
    train_linear_loader = Data.DataLoader(train_linear_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_dataset = Dataset(device=args.device, mode='test', data=Test_data, wave_len=args.wave_length)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    all_train_linear_dataset_with_image_name = Dataset_with_image_name(device=args.device, mode='supervise_train', data=Train_data_all_with_image_name, wave_len=args.wave_length)
    all_train_linear_loader_with_image_name = Data.DataLoader(all_train_linear_dataset_with_image_name, batch_size=args.train_batch_size)

    test_dataset_with_image_name = Dataset_with_image_name(device=args.device, mode='test', data=Test_data_with_image_name, wave_len=args.wave_length)
    test_loader_with_image_name = Data.DataLoader(test_dataset_with_image_name, batch_size=args.test_batch_size)

    print(args.data_shape)
    print('dataset initial ends')

    time_model = TimeEncoder(args)

    print('model initial ends')
    trainer = Trainer(args, time_model, train_loader, train_linear_loader, test_loader, verbose=True)

    train_mode = True # True for training, False for the export of test data for image generation

    if train_mode:
        # Select training stage (supports --train_stage CLI for ablation automation)
        stage = args.train_stage
        if stage == 'pretrain':
            trainer.pretrain()
        elif stage == 'finetune':
            trainer.finetune()
        elif stage == 'finetune_timefreq':
            trainer.finetune_timefreq()
        elif stage == 'finetune_CLIP':
            trainer.finetune_CLIP()
        elif stage == 'test':
            trainer.test()
        else:
            # Default: current hardcoded stage (for backward compatibility)
            trainer.finetune_CLIP()

    else:
        ## We suggest exporting data by single subject
        timeE = TimeEncoder(args).to("cuda")
        freq_model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value)
                              for (key, value) in [x.split("=") for x in opt.model_params]}
        # Create discriminator model
        freq_model = FreqEncoder(**freq_model_options)

        timefreq_model = TimeFreqEncoder(timeE, freq_model, args)
        timefreq_model = timefreq_model.to("cuda")

        freqtime_state_dict = torch.load(args.save_path + '/timefreqmodel.pkl', map_location="cuda")

        timefreq_model.load_state_dict(freqtime_state_dict)

        test_label,test_image_names, test_preds,test_seqs,test_rep,testacc=get_rep_with_label_with_image_name(timefreq_model,test_loader_with_image_name)

        all_train_label,train_image_names,all_train_preds,all_train_seqs,all_train_rep,trainacc= get_rep_with_label_with_image_name(timefreq_model, all_train_linear_loader_with_image_name) # 获取训练数据的模型编码和对应标签
        clf = fit_lr(all_train_rep, all_train_label)

        acc = clf.score(test_rep, test_label)
        pred_label = np.argmax(clf.predict_proba(test_rep), axis=1)

        if(acc>testacc):
            result = [[x] for x in pred_label]
            test_preds=result

        f1 = f1_score(test_label, test_preds, average='macro')
        print('acc:'+str(acc)+' f1:'+str(f1))

        import os
        from datetime import datetime

        # 定义目标路径
        save_dir = "exp/epilepsy/test/classification_acc"
        os.makedirs(save_dir, exist_ok=True)  # 自动创建目录（如果不存在）

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间戳格式化为字符串
        csv_filename = f"metrics_{timestamp}.csv"
        csv_file = os.path.join(save_dir, csv_filename)

        # 写入 CSV 文件
        with open(csv_file, "w") as f:
            f.write("Accuracy,F1_Score\n")          # 表头
            f.write(f"{acc},{f1}\n")              # 数据行

        torch.save(test_preds,'data/EEG_Feature_Label/test_pred.pth')
        torch.save(test_label, 'data/EEG_Feature_Label/test_label.pth')
        torch.save(test_image_names, 'data/EEG_Feature_Label/test_image_names.pth')
        torch.save(test_seqs, 'data/EEG_Feature_Label/test_seqs.pth')




if __name__ == '__main__':
    main()
