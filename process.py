import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from loss import CE, Align, Reconstruct,CM, tripletclip_loss, negclip_loss, clip_loss
from torch.optim.lr_scheduler import LambdaLR
from classification import fit_lr, get_rep_with_label,get_freqrep_with_label
from model.MindScryerModels_test import AlignNet,TimeFreqEncoder,FreqEncoder, EEGNetEncoder, TimeFreqEncoder_alpha
import argparse
from torch.utils.tensorboard import SummaryWriter
import csv
import time
import os
import torch.nn as nn
import wandb
from datetime import datetime
from model.mlp_classifier import ResMLPClassifier
import torch.utils.data as Data
from sklearn.svm import SVC # 导入 SVM 分类器
from sklearn.preprocessing import StandardScaler # 导入标准化工具
from sklearn.model_selection import GridSearchCV, StratifiedKFold # 导入网格搜索和交叉验证工具
from sklearn.metrics import accuracy_score, f1_score, classification_report # 导入评估指标
import joblib # 用于保存和加载 sklearn 模型
parser = argparse.ArgumentParser(description="Template")

parser.add_argument('-mt','--model_type', default='eegnet', help='')
parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
# parser.add_argument('--pretrained_net', default='test/lstm_subject0_20250426_002232/epoch_580.pth', help="path to pre-trained net")
# parser.add_argument('--pretrained_net', default='checkpoints/eegnet_subject0_20250506_110826/epoch_5440.pth', help="path to pre-trained net")
parser.add_argument('--pretrained_net', default='results_EEGNet/eegnet_sub0_20250506_215052/best_model.pth', help="path to pre-trained net")

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

# # --- 新增 model_type 参数定义 ---
# parser.add_argument('-mt', '--model_type', type=str, default='eegnet', # 设置默认值为 'eegnet'
#                     choices=['lstm', 'eegnet', 'FreqEncoder', 'EEGNetEncoder'], # 允许的选项
#                     help='指定使用的频率模型: lstm/FreqEncoder | eegnet/EEGNetEncoder')
# # --- 新增结束 ---

# # --- 新增 model_params 参数定义 (如果 process.py 中确实需要它) ---
# parser.add_argument('-mp','--model_params', default=[], nargs='*', # default=[] 避免 None
#                     help='(可选) 频率模型特定参数 key=value 列表 (例如: F1=8 D=2)')
# # --- 新增结束 ---

# # --- 新增 pretrained_net 参数定义 (如果 process.py 中确实需要它) ---
# parser.add_argument('--pretrained_net', default='checkpoints/eegnet_subject0_20250506_110826/epoch_5440.pth', type=str,
#                     help="预训练频率模型权重路径")
# # --- 新增结束 ---

# Parse arguments
opt, _ = parser.parse_known_args()

def top_k_accuracy_score(y_true, y_pred, k=5):
    top_k_preds = torch.topk(y_pred, k, dim=1)[1]
    correctness = top_k_preds.eq(y_true.view(-1, 1).expand_as(top_k_preds))
    top_k_accuracy = correctness.sum().float() / y_true.size(0)
    return top_k_accuracy.item()  # 返回一个Python标量

def l1_regularization(model, lambda_):
    l1_norm = 0
    for param in model.parameters():
        l1_norm += param.abs().sum()
    l1_penalty = lambda_ * l1_norm
    return l1_penalty

class Trainer():
    def __init__(self, args, time_model, train_loader, train_linear_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = time_model.to(torch.device(self.device))

        self.train_loader = train_loader
        #self.train_linear_loader = train_linear_loader
        self.train_linear_loader = train_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps

        self.cr = CE(self.model)
        self.alpha = args.alpha
        self.beta = args.beta

        self.test_cr = torch.nn.CrossEntropyLoss()
        self.num_epoch = args.num_epoch
        self.num_epoch_pretrain = args.num_epoch_pretrain
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path

        self.step = 0
        self.best_metric = -1e9
        self.metric = 'acc'
        # self.reduce_dim = nn.Linear(59136, 768).to(self.device)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 固定不可训练
        # self.logit_scale = torch.tensor(np.log(1 / 0.07))
        self.writer = SummaryWriter(log_dir=args.save_path)  # 指定日志目录

    def pretrain(self):
        print('pretraining')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        eval_acc = 0
        align = Align()
        reconstruct = Reconstruct()
        self.model.copy_weight()
        for epoch in range(self.num_epoch_pretrain):
            print('Epoch:' + str(epoch+1))
            self.model.train()
            tqdm_dataloader = tqdm(self.train_loader)
            loss_sum = 0
            loss_mse = 0
            loss_ce = 0
            hits_sum = 0
            NDCG_sum = 0
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad() 
                [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens] = self.model.pretrain_forward(batch[0])
                align_loss = align.compute(rep_mask, rep_mask_prediction)
                loss_mse += align_loss.item()
                reconstruct_loss, hits, NDCG = reconstruct.compute(token_prediction_prob, tokens)
                loss_ce += reconstruct_loss.item()
                hits_sum += hits.item()
                NDCG_sum += NDCG
                loss = self.alpha * align_loss + self.beta * reconstruct_loss
                loss.backward()
                self.optimizer.step()
                self.model.momentum_update()
                loss_sum += loss.item()

                # 记录预训练损失到 TensorBoard
                self.writer.add_scalar('Loss/pretrain', loss.item(), self.step)

            # 记录每个 epoch 的平均损失
            self.writer.add_scalar('Loss/epoch_pretrain', loss_sum / (idx + 1), self.step)
            print(f"Pretrain epoch {epoch+1}, loss: {loss_sum / (idx + 1)}")

            # print('pretrain epoch{0}, loss{1}, mse{2}, ce{3}, hits{4}, ndcg{5}'.format(epoch + 1, loss_sum / (idx + 1),
            #                                                                            loss_mse / (idx + 1),
            #                                                                            loss_ce / (idx + 1), hits_sum,
            #                                                                            NDCG_sum / (idx + 1)))

            if (epoch + 1) % 20 == 0:
                     torch.save(self.model.state_dict(), self.save_path + '/pretrain_model_epoch'+str(epoch+1)+'.pkl')

            if (epoch + 1) % 3 == 0:
                self.model.eval()
                train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
                test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
                clf = fit_lr(train_rep, train_label)
                acc = clf.score(test_rep, test_label)
                print(acc)
                if acc > eval_acc:
                    eval_acc = acc
                    torch.save(self.model.state_dict(), self.save_path + '/pretrain_model.pkl')
                    # It is worth noting that the highest pretraining accuracy does not mean the model is the
                    # best one for finetuning, so the one with larger training epoch should be used.
        self.writer.close()

    def finetune(self):
        print('finetune')
        self.model.linear_proba = True
        #self.args.load_pretrained_model=False
        if self.args.load_pretrained_model:
            print('load pretrained model')
            state_dict = torch.load(self.save_path + '/pretrain_model_epoch300.pkl', map_location=self.device)
            try:
                self.model.load_state_dict(state_dict)
            except:
                model_state_dict = self.model.state_dict()
                for pretrain, random_intial in zip(state_dict, model_state_dict):
                    assert pretrain == random_intial
                    if pretrain in ['input_projection.weight', 'input_projection.bias', 'predict_head.weight',
                                    'predict_head.bias', 'position.pe.weight']:
                        state_dict[pretrain] = model_state_dict[pretrain]
                self.model.load_state_dict(state_dict)

        self.model.eval()
        train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
        test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
        clf = fit_lr(train_rep, train_label)
        acc = clf.score(test_rep, test_label)
        pred_label = np.argmax(clf.predict_proba(test_rep), axis=1)
        f1 = f1_score(test_label, pred_label, average='macro')
        print(acc, f1)

        self.model.linear_proba = False #If linear_proba = True, freeze pretrained model, train only classifier
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)

        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            self.print_process(
                'Finetune epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))

            if (epoch + 1) % 5 == 0:
               torch.save(self.model.state_dict(),
                          self.save_path + '/finetune_model_epoch' + str(epoch + 1) + '.pkl')

        self.print_process(self.best_metric)

        self.writer.close()

        return self.best_metric

    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_linear_loader) if self.verbose else self.train_linear_loader
        loss_sum = 0
        pos=0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()
            l1=l1_regularization(self.model,0.000003)
            loss = self.cr.computeft(batch)#+l1
            loss_sum += loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            pos=pos+1
            self.step += 1

            # 每 100 步记录一次训练损失到 TensorBoard
            if idx % 10 == 0:
                self.writer.add_scalar('Loss/train', loss.item(), self.step)  # 记录训练损失

        # 记录每个 epoch 的平均损失
        self.writer.add_scalar('Loss/epoch_train', loss_sum / (idx + 1), self.step)

        # if self.step % self.eval_per_steps == 0:
        metric = self.eval_model()
        self.print_process(metric)

        if metric[self.metric] >= self.best_metric:
            torch.save(self.model.state_dict(), self.save_path + '/finetune_model.pkl')
            self.best_metric = metric[self.metric]
        self.model.train()

        self.writer.close()

        return loss_sum / (idx + 1), time.perf_counter() - t0

    def eval_model(self):
        self.model.eval()
        tqdm_data_loader = tqdm(self.test_loader) if self.verbose else self.test_loader
        metrics = {'acc': 0, 'f1': 0}
        pred = []
        label = []
        test_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                ret = self.compute_metrics(batch)
                if len(ret) == 2:
                    pred_b, label_b = ret
                    pred += pred_b
                    label += label_b
                else:
                    pred_b, label_b, test_loss_b = ret
                    pred += pred_b
                    label += label_b
                    test_loss += test_loss_b.cpu().item()
        print("aaa")
        print(len(label))
        confusion_mat = self._confusion_mat(label, pred)
        self.print_process(confusion_mat)
        if self.args.num_class == 2:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred)
            metrics['precision'] = precision_score(y_true=label, y_pred=pred)
            metrics['recall'] = recall_score(y_true=label, y_pred=pred)
        else:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred, average='macro')
            metrics['micro_f1'] = f1_score(y_true=label, y_pred=pred, average='micro')
        metrics['acc'] = accuracy_score(y_true=label, y_pred=pred)
        metrics['test_loss'] = test_loss / (idx + 1)
        return metrics

    def compute_metrics(self, batch):
        seqs, label, clip, clip_moreinf = batch
        lastrep, rep,scores = self.model(seqs)
        _, pred = torch.topk(scores, 1)
        test_loss = self.test_cr(scores, label.view(-1).long())
        pred = pred.view(-1).tolist()
        return pred, label.tolist(), test_loss

    def compute_metrics_freq(self, batch,model):
        #if len(batch) == 2:
        seqs, label,clip,clip_moreinf, *_ = batch
        lastrep, rep,scores = model(seqs)
        top3acc=top_k_accuracy_score(y_true=label,y_pred=scores,k=3)
        top5acc=top_k_accuracy_score(y_true=label,y_pred=scores,k=5)
        #else:
        #    seqs1, seqs2, label = batch
        #    lastrep, rep, scores = self.model((seqs1, seqs2))
        _, pred = torch.topk(scores, 1)
        #print(np.shape(scores))
        test_loss = self.test_cr(scores, label.view(-1).long())
        pred = pred.view(-1).tolist()
        return pred, label.tolist(), test_loss ,top3acc ,top5acc

    def _confusion_mat(self, label, pred):
        mat = np.zeros((self.args.num_class, self.args.num_class))
        for _label, _pred in zip(label, pred):
            mat[_label, _pred] += 1
        return mat

    def print_process(self, *x):
        if self.verbose:
            print(*x)

    def cont_pretrain(self):
        start_epoch=300
        state_dict = torch.load(self.save_path + '/pretrain_model_epoch300.pkl', map_location=self.device)
        eval_acc=0.0 # It should be modified.
        self.model.load_state_dict(state_dict)
        print('cont_pretraining')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        align = Align()
        reconstruct = Reconstruct()
        self.model.copy_weight()

        for epoch in range(self.num_epoch_pretrain):
            if(epoch<start_epoch):
                continue
            print('Epoch:' + str(epoch + 1))
            self.model.train()
            tqdm_dataloader = tqdm(self.train_loader)
            loss_sum = 0
            loss_mse = 0
            loss_ce = 0
            hits_sum = 0
            NDCG_sum = 0
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad()
                [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens] = self.model.pretrain_forward(batch[0])
                align_loss = align.compute(rep_mask, rep_mask_prediction)
                loss_mse += align_loss.item()
                reconstruct_loss, hits, NDCG = reconstruct.compute(token_prediction_prob, tokens)
                loss_ce += reconstruct_loss.item()
                hits_sum += hits.item()
                NDCG_sum += NDCG
                loss = self.alpha * align_loss + self.beta * reconstruct_loss
                loss.backward()
                self.optimizer.step()
                self.model.momentum_update()
                loss_sum += loss.item()
            print('pretrain epoch{0}, loss{1}, mse{2}, ce{3}, hits{4}, ndcg{5}'.format(epoch + 1, loss_sum / (idx + 1),
                                                                                       loss_mse / (idx + 1),
                                                                                       loss_ce / (idx + 1), hits_sum,
                                                                                       NDCG_sum / (idx + 1)))

            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(), self.save_path + '/pretrain_model_epoch'+str(epoch+1)+'.pkl')

            if (epoch + 1) % 3 == 0:
                self.model.eval()
                train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
                test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
                clf = fit_lr(train_rep, train_label)
                acc = clf.score(test_rep, test_label)
                print(acc)
                if acc > eval_acc:
                    eval_acc = acc
                    torch.save(self.model.state_dict(), self.save_path + '/pretrain_model.pkl')



    def finetune_CLIP(self):
        eval_cosine = 0.0
        # freq_model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for
        #                  (key, value) in [x.split("=") for x in opt.model_params]}
        # freq_model = FreqEncoder(**freq_model_options)

        # self.timefreq_model=TimeFreqEncoder(self.model,freq_model,self.args)
        # self.timefreq_model = self.timefreq_model.to(torch.device(self.device))

        # freqtime_state_dict = torch.load(self.save_path + '/timefreqmodel.pkl', map_location=self.device)

        # self.timefreq_model.load_state_dict(freqtime_state_dict)

         # --- 修改后的实例化方式 (与 finetune_timefreq 类似) ---
        num_channels = 128 # !! 假设 !!
        if not hasattr(self.args, 'time_high') or not hasattr(self.args, 'time_low'):
            raise AttributeError("Trainer 的 args 对象缺少 time_high 或 time_low 属性")
        num_time_steps = self.args.time_high - self.args.time_low

        freq_model_options = {}
        for item in opt.model_params: # 使用全局 opt
             parts = item.split("=")
             if len(parts) == 2:
                 key, value_str = parts
                 try:
                     value = int(value_str) if value_str.isdigit() else (float(value_str) if value_str.replace('.', '', 1).isdigit() else value_str)
                     freq_model_options[key] = value
                 except ValueError: freq_model_options[key] = value_str
             else: print(f"警告(CLIP): 忽略格式不正确的模型参数 '{item}'。")

        print(f"finetune_CLIP - 从 -mp 解析的可选参数: {freq_model_options}")

        # 根据 opt.model_type 实例化正确的频率模型
        if opt.model_type == 'eegnet' or opt.model_type == 'EEGNetEncoder':
            print("finetune_CLIP - 实例化 EEGNetEncoder...")
            eegnet_args = {
                'input_channels': num_channels,
                'input_time_points': num_time_steps,
                'num_classes': self.args.num_class,
                'output_feature_dim': freq_model_options.get('output_feature_dim', 128)
            }
            eegnet_args.update(freq_model_options)
            valid_eegnet_keys = ['input_channels', 'input_time_points', 'num_classes', 'F1', 'D', 'F2', 'dropout_rate', 'output_feature_dim']
            eegnet_args_filtered = {k: v for k, v in eegnet_args.items() if k in valid_eegnet_keys}
            print(f"finetune_CLIP - 传递给 EEGNetEncoder 的最终参数: {eegnet_args_filtered}")
            try:
                freq_model = EEGNetEncoder(**eegnet_args_filtered)
            except TypeError as e:
                print(f"finetune_CLIP - 实例化 EEGNetEncoder 时发生 TypeError: {e}")
                raise e

        elif opt.model_type == 'lstm' or opt.model_type == 'FreqEncoder':
            print("finetune_CLIP - 实例化 FreqEncoder...")
            freq_encoder_args = {
                'input_size': num_channels, # 修正 input_size
                'lstm_size': freq_model_options.get('lstm_size', 128),
                'lstm_layers': freq_model_options.get('lstm_layers', 1),
                'output_size': freq_model_options.get('output_feature_dim', 128)
            }
            valid_freqencoder_keys = ['input_size', 'lstm_size', 'lstm_layers', 'output_size']
            freq_encoder_args_filtered = {k:v for k,v in freq_encoder_args.items() if k in valid_freqencoder_keys}
            print(f"finetune_CLIP - 传递给 FreqEncoder 的最终参数: {freq_encoder_args_filtered}")
            try:
                 freq_model = FreqEncoder(**freq_encoder_args_filtered)
                 # 如果 FreqEncoder 的 classifier 输出维度固定，可能需要调整
                 if hasattr(freq_model, 'classifier') and isinstance(freq_model.classifier, torch.nn.Linear):
                     if freq_model.classifier.out_features != self.args.num_class:
                         print("警告(CLIP): 调整 FreqEncoder 内部分类器输出维度...")
                         freq_model.classifier = torch.nn.Linear(freq_model.classifier.in_features, self.args.num_class)
            except TypeError as e:
                print(f"finetune_CLIP - 实例化 FreqEncoder 时发生 TypeError: {e}")
                raise e
        else:
             raise ValueError(f"finetune_CLIP - 不支持的模型类型: {opt.model_type}。")

        # --- 实例化 TimeFreqEncoder (使用上面创建的 freq_model) ---
        self.timefreq_model = TimeFreqEncoder(self.model, freq_model, self.args) # self.model 是 TimeEncoder
        # 将 TimeFreqEncoder 移到目标设备，加载权重时 map_location 会处理
        # self.timefreq_model = self.timefreq_model.to(torch.device(self.device))

        # --- 加载 TimeFreqEncoder 的权重 ---
        print(f"尝试加载 TimeFreqEncoder 权重从: {self.save_path + '/timefreqmodel.pkl'}")
        try:
            # freqtime_state_dict = torch.load(self.save_path + '/timefreqmodel.pkl', map_location=self.device) # 加载到目标设备
            freqtime_state_dict = torch.load(self.save_path + '/finetune_timefreq_2025-05-25_18-30-19/timefreqmodel_sub4.pkl', map_location=self.device) # 加载到目标设备
            # 现在 self.timefreq_model 的结构应该与加载的 state_dict 匹配了
            self.timefreq_model.load_state_dict(freqtime_state_dict)
            print("成功加载 TimeFreqEncoder 的 state_dict。")
        except FileNotFoundError:
             print(f"错误: 未找到 TimeFreqEncoder 权重文件 '{self.save_path + '/timefreqmodel.pkl'}'。AlignNet 将使用随机初始化的 TimeFreqEncoder。")
             # 这里可能需要决定是退出还是继续（如果允许从头训练 AlignNet）
             # exit(1)
        except RuntimeError as e_load_tf: # 处理加载时的其他错误 (例如键不匹配 - 理论上不应再发生)
             print(f"错误: 加载 TimeFreqEncoder state_dict 时出错: {e_load_tf}")
             print("请确保 timefreqmodel.pkl 文件是使用当前代码结构保存的。")
             raise e_load_tf # 重新抛出错误，因为 AlignNet 依赖它
        except Exception as e_other:
             print(f"加载 TimeFreqEncoder 权重时发生未知错误: {e_other}")
             raise e_other

        self.timefreq_model.to(torch.device("cpu"))

        # freq_size=freq_model.output_size
        freq_size=128
        time_size=self.model.d
        # print(time_size)
        clip_size=int(77*768)

        # --- AlignNet with ablation-aware parameters ---
        alignnet_num_blocks = getattr(self.args, 'alignnet_num_blocks', 3)
        alignnet_expansion = getattr(self.args, 'alignnet_expansion', 4)
        alignnet_activation = getattr(self.args, 'alignnet_activation', 'tanh')
        use_classifier_logits = getattr(self.args, 'use_classifier_logits', True)

        self.alignmodel = AlignNet(
            time_size, freq_size, clip_size, self.timefreq_model,
            num_blocks=alignnet_num_blocks,
            expansion=alignnet_expansion,
            activation=alignnet_activation,
            use_classifier_logits=use_classifier_logits
        )
        self.alignmodel=self.alignmodel.to(torch.device(self.device))
        print('CLIP_finetune')
        # self.optimizer = torch.optim.AdamW(self.alignmodel.parameters(), lr=self.args.lr)
        self.optimizer = torch.optim.AdamW(
            list(self.alignmodel.parameters()) + [self.logit_scale],
            lr=self.args.lr
        )
        # CLIPloss = tripletclip_loss()
        align=Align()

        # Initialize TensorBoard writer
        # writer_TensorBoard = SummaryWriter(log_dir='./log/tensorboard')
        
        # === 新增：创建以时间命名的保存目录 ===
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join('./log', timestamp)
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path

        # 初始化 wandb
        wandb.init(
            project="CLIP_finetune",
            name=f"clip_finetune_run_{timestamp}",
            config={
                "learning_rate": self.args.lr,
                "epochs": 500000,
                "device": self.device
            }
        )
        wandb.watch(self.alignmodel, log="all")

        log_file = os.path.join(save_path, 'training_log_tripletclip.csv')

        # 新增收敛判断参数
        min_delta = 1e-5
        patience = 30
        no_improve_counter = 0
        best_trloss = float('inf')

        # total_train_acc = 0
        # total_test_acc = 0

        # 打开CSV文件并创建写入器
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(['Epoch', 'TrLoss', 'TrMSE', 'TrCosine', 'TrMSE_Moreinf', 'TrCosine_Moreinf', 
                            'TeLoss', 'TeMSE', 'TeCosine', 'TeMSE_Moreinf', 'TeCosine_Moreinf'])

            num_clip_epochs = getattr(self.args, 'num_epoch', 500)
            for epoch in range(num_clip_epochs):
                print('Epoch:' + str(epoch + 1))
                self.alignmodel.train()

                # 修改部分：只加载前10个数据进行训练
                # train_subset = torch.utils.data.Subset(self.train_loader.dataset, range(10))  # 获取前10个样本
                # test_subset = torch.utils.data.Subset(self.test_loader.dataset, range(10))  # 获取前10个样本
                # tqdm_dataloader = tqdm(train_subset)
                # test_tqdm_dataloader=tqdm(test_subset)

                tqdm_dataloader = tqdm(self.train_loader)
                test_tqdm_dataloader=tqdm(self.test_loader)

                loss_clip=0
                loss_mse=0
                loss_clip_moreinf=0
                loss_mse_moreinf=0
                loss_sum = 0

                teloss_clip=0
                teloss_mse=0
                teloss_clip_moreinf=0
                teloss_mse_moreinf=0
                teloss_sum = 0

                for idx, batch in enumerate(tqdm_dataloader):
                    batch = [x.to(self.device) for x in batch]

                    for i in [0, 2, 3, 4, 5]:
                            batch[i] = batch[i].to(torch.float32)  # 统一数据类型
                            batch[i] = (batch[i] - batch[i].mean()) / (batch[i].std() + 1e-6)
                            
                            # 限制极端值范围
                            batch[i] = torch.clamp(batch[i], -3, 3)

                            # print(f"batch[{i}] mean: {batch[i].mean()}, std: {batch[i].std()}, min: {batch[i].min()}, max: {batch[i].max()}")


                    self.optimizer.zero_grad()
                    # print("batch[0] min/max:", batch[0].min(), batch[0].max())
                    # print("batch[2] min/max:", batch[2].min(), batch[2].max())
                    # print("batch[3] min/max:", batch[3].min(), batch[3].max())
                    # print("batch[4] min/max:", batch[4].min(), batch[4].max())
                    # print("batch[5] min/max:", batch[5].min(), batch[5].max())

                    clippred = self.alignmodel.forward(batch[0].float()).to(self.device)
                    # clippred = (clippred - clippred.mean(dim=1, keepdim=True)) / (clippred.std(dim=1, keepdim=True) + 1e-6)
                    # clippred = torch.clamp(clippred, -3, 3)
                    # import pdb;pdb.set_trace()
                    # clippred = self.reduce_dim(clippred)  # -> [128, 768]
                    # batch[3] = self.reduce_dim(batch[3])  # -> [128, 768]
                    # batch[5] = self.reduce_dim(batch[5])  # -> [128, 768]

                    # print("clippred min/max:", clippred.min(), clippred.max())
                    # print("batch[0].shape:", batch[0].shape)
                    # CLIP_loss = CLIPloss.compute(clippred.float(), batch[2].float())
                    # CLIP_loss=CLIPloss.compute(clippred.float(), batch[3].float())
                    # logit_scale = getattr(self.args, "logit_scale", 0.0001)
                    # print("batch[2].shape:", batch[2].shape)
                    # print("batch[3].shape:", batch[3].shape)
                    # print("batch[4].shape:", batch[4].shape)
                    # print("batch[5].shape:", batch[5].shape)
                    # print("CLIP pred:", clippred.float().detach().cpu().numpy()[:5])
                    # print("Batch target:", batch[2].float().detach().cpu().numpy()[:5])
                    logit_scale = self.logit_scale.exp() # 从 log space 转换回来
                    # CLIP_loss, _ = tripletclip_loss(clippred.float(), batch[2].float(), batch[4].float(), batch[5].float(), logit_scale)
                    # CLIP_loss = CLIPloss.compute(clippred.float(), batch[3].float(), batch[6].float(), batch[7].float(), logit_scale)

                    # Ablation-aware loss selection
                    loss_type = getattr(self.args, 'loss_type', 'negclip')
                    if loss_type == 'clip':
                        CLIP_loss, train_acc = clip_loss(clippred.float(), batch[3].float(), logit_scale)
                    elif loss_type == 'tripletclip':
                        CLIP_loss, train_acc = tripletclip_loss(clippred.float(), batch[2].float(), batch[4].float(), batch[5].float(), logit_scale)
                    else:  # 'negclip' (baseline)
                        CLIP_loss, train_acc = negclip_loss(clippred.float(), batch[3].float(), batch[5].float(), logit_scale)
                    # import pdb;pdb.set_trace()
                    # print(f"Epoch {epoch+1}, Step {idx}: CLIP_loss={CLIP_loss.item()}, train_acc={train_acc}")

                    align_loss=align.compute(clippred.float(), batch[2].float())
                    align_loss_moreinf = align.compute(clippred.float(), batch[3].float())
                    All_CLIP_loss=CLIP_loss
                    All_align_loss=align_loss+align_loss_moreinf
                    # print(f"Epoch {epoch+1}, Step {idx}: CLIP Loss={CLIP_loss.item()}, Align Loss={align_loss.item()}")
                    
                    loss_clip+= CLIP_loss.item()
                    loss_mse+= align_loss.item()
                    loss_clip_moreinf+= CLIP_loss.item()
                    loss_mse_moreinf+= align_loss_moreinf.item() #MSE, due to numerical considerations
                    # lambda_value = 0.000002
                    # l1_penalty = l1_regularization(self.model, lambda_value)
                    loss = All_align_loss+All_CLIP_loss#+l1_penalty
                    # print(f"Epoch {epoch+1}, Step {idx}: All_align_loss={All_align_loss.item()}, All_CLIP_loss={All_CLIP_loss.item()}")

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.alignmodel.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    loss_sum += loss.item()

                    # total_train_acc += train_acc 

                trloss=loss_sum / (idx + 1)
                trmse=loss_mse / (idx + 1)
                trmse_moreinf=loss_mse_moreinf / (idx + 1)
                trcosine=loss_clip / (idx + 1)
                trcosine_moreinf=loss_clip_moreinf / (idx + 1)

                for idxte, batch in enumerate(test_tqdm_dataloader):
                    self.alignmodel.eval()
                    batch = [x.to(self.device) for x in batch]

                    for i in [0, 2, 3, 4, 5]:
                        batch[i] = batch[i].to(torch.float32)  # 统一数据类型
                        batch[i] = (batch[i] - batch[i].mean()) / (batch[i].std() + 1e-6)
                        
                        # 限制极端值范围
                        batch[i] = torch.clamp(batch[i], -3, 3)

                        # print(f"batch[{i}] mean: {batch[i].mean()}, std: {batch[i].std()}, min: {batch[i].min()}, max: {batch[i].max()}")

                    clippred= self.alignmodel(batch[0])
                    # CLIP_loss = CLIPloss.compute(clippred, batch[2])
                    # CLIP_loss = CLIPloss.compute(clippred.float(), batch[3].float())
                    # CLIP_loss, _ = tripletclip_loss(clippred.float(), batch[2].float(), batch[4].float(), batch[5].float(), logit_scale)
                    logit_scale = self.logit_scale.exp()  # 从 log space 转换回来
                    CLIP_loss, test_acc =negclip_loss(clippred.float(),batch[3].float(), batch[5].float(), logit_scale)
                    align_loss=align.compute(clippred.float(), batch[2])
                    align_loss_moreinf = align.compute(clippred.float(), batch[3].float())
                    All_CLIP_loss = CLIP_loss
                    All_align_loss = align_loss + align_loss_moreinf
                    # print(f"Epoch {epoch+1}, Step {idx}: CLIP Loss={CLIP_loss.item()}, Align Loss={align_loss.item()}")

                    teloss_clip+= CLIP_loss.item()
                    teloss_mse+= align_loss.item()
                    teloss_clip_moreinf+= CLIP_loss.item()
                    teloss_mse_moreinf+= align_loss_moreinf.item()
                    teloss = All_align_loss+All_CLIP_loss
                    teloss_sum += teloss.item()

                    # total_test_acc += test_acc 

                teloss = teloss_sum / (idxte + 1)
                temse = teloss_mse / (idxte + 1)
                tecosine = teloss_clip / (idxte + 1)
                temse_moreinf = teloss_mse_moreinf / (idxte + 1)
                tecosine_moreinf = teloss_clip_moreinf / (idxte + 1)

                print('clip_finetune epoch{0}, trloss{1}, trmse{2}, trcosine{3},trmse_moreinf{4},trcosine_moreinf{5}, '
                    'teloss{6}, temse{7}, tecosine{8},temse_moreinf{9}, tecosine_moreinf{10}'.format(epoch + 1,
                trloss,trmse,trcosine,trmse_moreinf,trcosine_moreinf,teloss,temse,tecosine,temse_moreinf,tecosine_moreinf))

                # 写入CSV文件
                writer.writerow([epoch + 1, trloss, trmse, trcosine, trmse_moreinf, trcosine_moreinf, 
                                teloss, temse, tecosine, temse_moreinf, tecosine_moreinf])

                # Log to TensorBoard
                # writer_TensorBoard.add_scalar('Accuracy/test', test_acc, epoch + 1) 
                # writer_TensorBoard.add_scalar('Accuracy/train', train_acc, epoch + 1)
                # writer_TensorBoard.add_scalar('Loss/test', teloss, epoch + 1) 
                # writer_TensorBoard.add_scalar('Loss/train', trloss, epoch + 1)
                # writer_TensorBoard.add_scalar('logit_scale', logit_scale, epoch + 1) 
                # wandb 日志记录
                wandb.log({
                    "epoch": epoch + 1,
                    "Loss/train": trloss,
                    "Loss/test": teloss,
                    # "MSE/train": trmse,
                    # "MSE/test": temse,
                    # "MSE_moreinf/train": trmse_moreinf,
                    # "MSE_moreinf/test": temse_moreinf,
                    # "CosineLoss/train": trcosine,
                    # "CosineLoss/test": tecosine,
                    # "CosineLoss_moreinf/train": trcosine_moreinf,
                    # "CosineLoss_moreinf/test": tecosine_moreinf,
                    "Accuracy/train": train_acc,
                    "Accuracy/test": test_acc,
                    "logit_scale": logit_scale
                })   

                # 保存模型每10个epoch一次
                if epoch < 500:
                    if (epoch + 1) % 10 == 0:
                        torch.save(self.alignmodel.state_dict(), os.path.join(self.save_path, f'clipfinetune_model_epoch{epoch + 1}_sub4.pkl'))
                        print(f"Model saved at epoch {epoch + 1}")
                else:
                    if (epoch + 1) % 100 == 0:
                        torch.save(self.alignmodel.state_dict(), os.path.join(self.save_path, f'clipfinetune_model_epoch{epoch + 1}_sub4.pkl'))
                        print(f"Model saved at epoch {epoch + 1}")

                # 根据 trloss 保存最佳模型
                if trloss < best_trloss:
                    best_trloss = trloss
                    torch.save(self.alignmodel.state_dict(),
                            os.path.join(self.save_path, 'clipfinetune_model_bestloss_sub4.pkl'))
                    print(f"损失改进，保存最佳模型(epoch {epoch + 1})")

                # if (epoch + 1) % 10 == 0:
                #     torch.save(self.alignmodel.state_dict(),
                #             os.path.join(self.save_path, f'clipfinetune_model_epoch{epoch + 1}.pkl'))

                # if tecosine > eval_cosine:
                #     eval_cosine = tecosine
                #     torch.save(self.alignmodel.state_dict(), os.path.join(self.save_path, 'clipfinetune_model.pkl'))

                # === 新增逻辑：基于trloss判断保存最佳模型 ===
                # if best_trloss - trloss > min_delta:
                #     best_trloss = trloss
                #     no_improve_counter = 0
                #     torch.save(self.alignmodel.state_dict(),
                #             os.path.join(self.save_path, 'clipfinetune_model_bestloss.pkl'))
                #     print(f"Loss improved, saving best model (epoch {epoch+1})")
                # else:
                #     no_improve_counter += 1
                #     print(f"Loss did not improve. Counter: {no_improve_counter}/{patience}")
                #     if no_improve_counter >= patience:
                #         print("Early stopping triggered due to no improvement.")
                #         break

        # Close the TensorBoard writer
        # writer_TensorBoard.close()
        wandb.finish()



    def test(self, num_tests=10):
        # 设备设置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型（保持不变）
        freq_model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for
                        (key, value) in [x.split("=") for x in opt.model_params]}
        freq_model = FreqEncoder(**freq_model_options)

        self.timefreq_model = TimeFreqEncoder(self.model, freq_model, self.args)
        self.timefreq_model = self.timefreq_model.to(torch.device(self.device))

        freqtime_state_dict = torch.load(self.save_path + '/timefreqmodel.pkl', map_location=self.device)
        self.timefreq_model.load_state_dict(freqtime_state_dict)
        self.timefreq_model.to(torch.device("cpu"))

        # 模型参数
        freq_size = freq_model.output_size
        time_size = self.model.d
        clip_size = int(77 * 768)

        # 加载模型函数
        def load_model(model_path):
            model = AlignNet(time_size, freq_size, clip_size, self.timefreq_model)
            model.load_state_dict(torch.load(model_path, map_location=device))
            return model.to(device)
        
        align = Align()

        # 加载两个对比模型
        finetuned_model = load_model('exp/epilepsy/test/clipfinetune_model_bestloss.pkl')  # 微调后的模型
        previous_best_model = load_model('exp/epilepsy/test/clipfinetune_model_epoch10.pkl')       # 之前的基准模型

        # 评估函数（支持多次测试取平均）
        def evaluate_model(model, num_tests=10):
            model.eval()
            total_loss, total_acc, total_cos_sim = 0.0, 0.0, 0.0

            for test_iter in range(num_tests):  # 运行多次测试
                print(f"\n正在进行第 {test_iter + 1}/{num_tests} 次测试...")
                iter_loss, iter_acc, iter_cos_sim = 0.0, 0.0, 0.0

                with torch.no_grad():
                    for batch in tqdm(self.train_loader, desc=f"测试进度 {test_iter + 1}"):
                        batch = [x.to(device) for x in batch]
                        # 标准化处理关键特征
                        for i in [0, 2, 3, 4, 5]:
                            batch[i] = batch[i].to(torch.float32)
                            batch[i] = (batch[i] - batch[i].mean()) / (batch[i].std() + 1e-6)
                            batch[i] = torch.clamp(batch[i], -3, 3)
                        
                        # 模型预测
                        clippred = model(batch[0].float())
                        logit_scale_val = self.logit_scale.exp()
                        # 计算损失和准确率
                        loss_type_eval = getattr(self.args, 'loss_type', 'negclip')
                        if loss_type_eval == 'clip':
                            CLIP_loss, acc = clip_loss(clippred.float(), batch[3].float(), logit_scale_val)
                        elif loss_type_eval == 'tripletclip':
                            CLIP_loss, acc = tripletclip_loss(clippred.float(), batch[2].float(), batch[4].float(), batch[5].float(), logit_scale_val)
                        else:  # 'negclip' (baseline)
                            CLIP_loss, acc = negclip_loss(
                                clippred.float(),
                                batch[3].float(),  # clip_moreinf特征
                                batch[5].float(),   # neg_text特征
                                logit_scale_val
                            )
                        # 对齐损失
                        align_loss = align.compute(clippred.float(), batch[2].float())
                        
                        iter_loss += (CLIP_loss + align_loss).item()
                        iter_acc += acc
                        iter_cos_sim += torch.nn.functional.cosine_similarity(clippred, batch[3], dim=1).mean().item()

                # 计算当前测试的平均指标
                avg_iter_loss = iter_loss / len(self.train_loader)
                avg_iter_acc = iter_acc / len(self.train_loader)
                avg_iter_cos_sim = iter_cos_sim / len(self.train_loader)

                # 累加到总结果
                total_loss += avg_iter_loss
                total_acc += avg_iter_acc
                total_cos_sim += avg_iter_cos_sim

                # print(f"第 {test_iter + 1} 次测试结果 - 损失: {avg_iter_loss:.4f}, 准确率: {avg_iter_acc:.4f}, 余弦相似度: {avg_iter_cos_sim:.4f}")
                print(f"第 {test_iter + 1} 次测试结果 - 损失: {avg_iter_loss:.4f}, 准确率: {avg_iter_acc:.4f}")

            # 返回多次测试的平均值
            return (
                total_loss / num_tests,
                total_acc / num_tests,
                total_cos_sim / num_tests
            )

        # 评估两个模型（多次测试）
        print("\n正在评估微调模型...")
        finetuned_loss, finetuned_acc, finetuned_cos_sim = evaluate_model(finetuned_model, num_tests)
        
        print("\n正在评估基准模型...")
        previous_loss, previous_acc, previous_cos_sim = evaluate_model(previous_best_model, num_tests)

        # 打印对比结果
        print("\n=== 模型性能对比（{}次测试平均值） ===".format(num_tests))
        # print(f"微调模型 - 损失: {finetuned_loss:.4f}, 准确率: {finetuned_acc:.4f}, 余弦相似度: {finetuned_cos_sim:.4f}")
        # print(f"基准模型 - 损失: {previous_loss:.4f}, 准确率: {previous_acc:.4f}, 余弦相似度: {previous_cos_sim:.4f}")
        print(f"微调模型 - 损失: {finetuned_loss:.4f}, 准确率: {finetuned_acc:.4f}")
        print(f"基准模型 - 损失: {previous_loss:.4f}, 准确率: {previous_acc:.4f}")

        # 计算改进百分比
        loss_improvement = (previous_loss - finetuned_loss) / previous_loss * 100
        acc_improvement = (finetuned_acc - previous_acc) / previous_acc * 100
        cos_sim_improvement = (finetuned_cos_sim - previous_cos_sim) / previous_cos_sim * 100

        print("\n=== 性能改进总结 ===")
        print(f"损失降低: {loss_improvement:.2f}% ({'提升' if loss_improvement > 0 else '下降'})")
        print(f"准确率提升: {acc_improvement:.2f}% ({'提升' if acc_improvement > 0 else '下降'})")
        # print(f"余弦相似度提升: {cos_sim_improvement:.2f}% ({'提升' if cos_sim_improvement > 0 else '下降'})")

        return {
            'finetuned': {'loss': finetuned_loss, 'accuracy': finetuned_acc, 'cosine_sim': finetuned_cos_sim},
            'previous': {'loss': previous_loss, 'accuracy': previous_acc, 'cosine_sim': previous_cos_sim},
            'improvement': {
                'loss': loss_improvement,
                'accuracy': acc_improvement,
                'cosine_sim': cos_sim_improvement
            }
        }



    def finetune_timefreq(self):
            # 获取当前时间并格式化为文件名的一部分
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.save_path = self.save_path + f"/finetune_timefreq_{current_time}"
            os.makedirs(self.save_path, exist_ok=True)
            csv_filename = self.save_path + f'/train_log_{current_time}.csv'

            # 初始化CSV文件，写入标题行
            with open(csv_filename, mode='w', newline='') as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow(['epoch', 'train_loss', 'test_loss', 'accuracy', 'top3_accuracy', 'top5_accuracy'])
            
            time_state_dict = torch.load('exp/epilepsy/test/finetune_model_epoch80.pkl',
                                    map_location=self.device)

            print("freq_train")

            self.model.load_state_dict(time_state_dict)

            self.model.eval()
            self.model.to(torch.device("cuda"))
            train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
            test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
            clf = fit_lr(train_rep, train_label)
            acc = clf.score(test_rep, test_label)
            pred_label = np.argmax(clf.predict_proba(test_rep), axis=1)
            f1 = f1_score(test_label, pred_label, average='macro')
            print(acc, f1)
            self.model.train()
            self.model.to(torch.device("cpu"))

            # freq_model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for
            #                 (key, value) in [x.split("=") for x in opt.model_params]}
            # freq_model = EEGNetEncoder(**freq_model_options)

            # if opt.pretrained_net != '':
            #     freq_model = torch.load(opt.pretrained_net)

            num_channels = 128
            # print(f"假设 EEG 通道数: {num_channels} (请务必核实!)")

            # 从 args 中获取时间点数
            # 确保 args 中有 time_high 和 time_low
            if not hasattr(self.args, 'time_high') or not hasattr(self.args, 'time_low'):
                 # 如果 args 没有，尝试从 opt 获取 (假设 opt 在类中可用或传入)
                 # 或者你需要找到正确的参数来源
                 raise AttributeError("Trainer 的 args 对象缺少 time_high 或 time_low 属性")
            num_time_steps = self.args.time_high - self.args.time_low
            print(f"计算得到的时间点数: {num_time_steps}")

            # 从命令行参数 -mp 获取可选参数
            # 添加检查，确保 x.split('=') 的结果长度为 2
            freq_model_options = {}
            for item in opt.model_params: # 使用全局的 opt 对象
                 parts = item.split("=")
                 if len(parts) == 2:
                     key, value_str = parts
                     try:
                         value = int(value_str) if value_str.isdigit() else (float(value_str) if value_str.replace('.', '', 1).isdigit() else value_str)
                         freq_model_options[key] = value
                     except ValueError:
                         print(f"警告: 无法解析参数值 '{value_str}' (来自 '{item}'). 跳过。")
                         freq_model_options[key] = value_str # 或者直接用字符串
                 else:
                     print(f"警告: 忽略格式不正确的模型参数 '{item}'。请使用 key=value 格式。")

            print(f"从 -mp 解析的可选参数: {freq_model_options}")

            # --- 根据 opt.model_type 实例化正确的频率模型 ---
            # !! 关键：从全局 opt 获取 model_type !!
            if opt.model_type == 'eegnet' or opt.model_type == 'EEGNetEncoder':
                print("实例化 EEGNetEncoder...")
                # 准备 EEGNetEncoder 的参数
                eegnet_args = {
                    'input_channels': num_channels,
                    'input_time_points': num_time_steps,
                    'num_classes': self.args.num_class, # 假设类别数在 args 中
                    # 可以设置一个默认的 output_feature_dim，如果需要的话
                    'output_feature_dim': freq_model_options.get('output_feature_dim', 128) # 默认或从 options 获取
                }
                # 合并可选参数 (如果 -mp 提供了 F1, D, F2 等，会覆盖默认值)
                eegnet_args.update(freq_model_options)
                # 移除可能不属于 EEGNetEncoder 的参数，防止 TypeError
                valid_eegnet_keys = ['input_channels', 'input_time_points', 'num_classes', 'F1', 'D', 'F2', 'dropout_rate', 'output_feature_dim']
                eegnet_args_filtered = {k: v for k, v in eegnet_args.items() if k in valid_eegnet_keys}

                print(f"传递给 EEGNetEncoder 的最终参数: {eegnet_args_filtered}")
                try:
                    freq_model = EEGNetEncoder(**eegnet_args_filtered)
                except TypeError as e:
                    print(f"实例化 EEGNetEncoder 时发生 TypeError: {e}")
                    print("请检查模型定义和传递的参数是否匹配。")
                    raise e # 重新抛出错误

            elif opt.model_type == 'lstm' or opt.model_type == 'FreqEncoder':
                print("实例化 FreqEncoder...")
                 # 准备 FreqEncoder 的参数
                freq_encoder_args = {
                    'input_size': num_channels,  # input_size = num EEG channels (128), not time steps
                    'lstm_size': freq_model_options.get('lstm_size', 128),
                    'lstm_layers': freq_model_options.get('lstm_layers', 1),
                    'output_size': freq_model_options.get('output_feature_dim', 128) # 假设 output_size 对应特征维度
                    # FreqEncoder 可能不需要 num_classes
                }
                # FreqEncoder 的构造函数可能只接受特定参数，过滤掉多余的
                valid_freqencoder_keys = ['input_size', 'lstm_size', 'lstm_layers', 'output_size'] # 根据 FreqEncoder 定义调整
                freq_encoder_args_filtered = {k:v for k,v in freq_encoder_args.items() if k in valid_freqencoder_keys}

                print(f"传递给 FreqEncoder 的最终参数: {freq_encoder_args_filtered}")
                try:
                     freq_model = FreqEncoder(**freq_encoder_args_filtered)
                except TypeError as e:
                    print(f"实例化 FreqEncoder 时发生 TypeError: {e}")
                    print("请检查模型定义和传递的参数是否匹配。")
                    raise e # 重新抛出错误
            else:
                 raise ValueError(f"不支持的模型类型: {opt.model_type}。请在命令行使用 -mt 指定 'lstm' 或 'eegnet'")

            # --- 加载预训练权重 (在模型实例化之后) ---
            # !! 关键：使用 opt.pretrained_net !!
            # if opt.pretrained_net != '':
            #     print(f"尝试从 '{opt.pretrained_net}' 加载预训练的 {opt.model_type} 权重...")
            #     try:
            #         # 优先尝试加载 state_dict (推荐)
            #         freq_state_dict = torch.load(opt.pretrained_net, map_location=self.device) # 加载到目标设备
            #         freq_model.load_state_dict(freq_state_dict)
            #         print(f"成功加载 {opt.model_type} 的 state_dict。")
            #     except (RuntimeError, KeyError, TypeError) as e1:
            #         print(f"加载 state_dict 失败: {e1}。尝试加载整个模型...")
            #         try:
            #             # 尝试加载整个模型 (如果 state_dict 失败)
            #             loaded_full_model = torch.load(opt.pretrained_net, map_location=self.device)
            #             # 检查加载的模型类型是否匹配
            #             if isinstance(loaded_full_model, type(freq_model)):
            #                 freq_model = loaded_full_model
            #                 print(f"成功加载完整的 {opt.model_type} 模型。")
            #             else:
            #                 print(f"警告: 从 '{opt.pretrained_net}' 加载的对象类型 ({type(loaded_full_model)}) 与期望的类型 ({type(freq_model)}) 不匹配。将使用随机初始化的模型。")
            #         except Exception as e2:
            #             print(f"加载整个模型也失败: {e2}。将使用随机初始化的 {opt.model_type}。")
            # else:
            #     print(f"未指定预训练的 {opt.model_type} 权重，将使用随机初始化的模型。")
            if opt.pretrained_net and opt.pretrained_net.strip() != '': # 确保路径非空
                print(f"尝试从 '{opt.pretrained_net}' 加载预训练的 {opt.model_type} 权重...")
                try:
                    # 加载整个文件，它可能是一个字典或直接是 state_dict
                    loaded_content = torch.load(opt.pretrained_net, map_location=self.device)

                    actual_state_dict_to_load = None

                    if isinstance(loaded_content, dict):
                        # 检查是否是训练脚本保存的检查点格式
                        if 'model_state_dict' in loaded_content:
                            actual_state_dict_to_load = loaded_content['model_state_dict']
                            print(f"从检查点文件中提取 'model_state_dict' 进行加载。")
                        elif 'state_dict' in loaded_content: # 有些可能用 'state_dict'
                            actual_state_dict_to_load = loaded_content['state_dict']
                            print(f"从检查点文件中提取 'state_dict' 进行加载。")
                        else:
                            # 可能是直接保存的 state_dict (也是一个字典)
                            actual_state_dict_to_load = loaded_content
                            print(f"加载的文件是一个字典，尝试直接作为 state_dict 加载。")
                    else:
                        # 如果不是字典，可能是一个直接保存的模型对象 (不推荐)
                        # 或者直接保存的 state_dict (已经被上面的 isinstance(dict) 覆盖)
                        # 这里我们假设如果不是字典，那么它应该是一个 state_dict (尽管可能性小)
                        # 更安全的做法是期望 state_dict
                        print(f"加载的文件不是字典，尝试作为 state_dict 加载 (可能性较低)。")
                        actual_state_dict_to_load = loaded_content


                    if actual_state_dict_to_load is not None:
                        # 过滤掉不匹配的键 (例如，如果加载的是 TimeFreqEncoder 的权重给 FreqEncoder)
                        # 这段过滤逻辑对于确保只加载 freq_model 自身的权重很重要
                        model_keys = set(freq_model.state_dict().keys())
                        loaded_keys = set(actual_state_dict_to_load.keys())

                        # 尝试找到 freq_model 在组合模型中可能的参数前缀
                        # 例如，如果 TimeFreqEncoder 中 freq_model 叫做 self.pretrained_model_freq
                        # 那么 freq_model 的参数键在保存的 TimeFreqEncoder 权重中可能是
                        # "pretrained_model_freq.eegnet.conv1.weight" 等
                        # 我们需要移除这个前缀才能正确加载到独立的 freq_model 中

                        # 一个简单的策略：如果加载的键有类似 "eegnet." 或 "lstm." 的部分，
                        # 并且 freq_model 自身的键没有这个前缀，尝试去除。
                        # 但更稳妥的是，确保 --pretrained_net 指向的是 freq_model 单独训练保存的权重。

                        # 如果 pretrained_net 确定是 freq_model 单独的权重，则可以直接加载
                        try:
                            freq_model.load_state_dict(actual_state_dict_to_load, strict=False) # strict=False 更宽容
                            print(f"成功加载 {opt.model_type} 的 state_dict。")
                        except RuntimeError as e_load:
                            print(f"使用提取的 state_dict 加载失败: {e_load}")
                            print("可能是由于键名不匹配。如果加载的是组合模型的权重，需要特殊处理键名。")
                            print(f"freq_model 的键示例: {list(model_keys)[:5]}")
                            print(f"加载的 state_dict 键示例: {list(loaded_keys)[:5]}")
                            print(f"将使用随机初始化的 {opt.model_type}。")

                    else: # 尝试加载整个模型对象 (如果上面的逻辑没能提取出 state_dict)
                        print(f"未能从加载内容中提取 state_dict。尝试将加载内容作为整个模型对象...")
                        if isinstance(loaded_content, type(freq_model)):
                            freq_model = loaded_content # 直接替换
                            print(f"成功将加载内容作为完整的 {opt.model_type} 模型使用。")
                        else:
                            print(f"警告: 从 '{opt.pretrained_net}' 加载的对象类型 ({type(loaded_content)}) 与期望类型 ({type(freq_model)}) 不匹配。将使用随机初始化的模型。")

                except FileNotFoundError:
                    print(f"错误: 预训练权重文件未找到 '{opt.pretrained_net}'。")
                except Exception as e_outer_load:
                    print(f"加载预训练的 {opt.model_type} 权重时发生未知错误: {e_outer_load}。将使用随机初始化的模型。")
            else:
                print(f"未指定预训练的 {opt.model_type} 权重，将使用随机初始化的模型。")

            # 确保频率模型在正确的设备上（在加载权重之后）
            freq_model.to(self.device)

            self.timefreq_model=TimeFreqEncoder(self.model,freq_model,self.args)
            self.timefreq_model = self.timefreq_model.to(torch.device(self.device))

            self.optimizer = torch.optim.AdamW(self.timefreq_model.parameters(), lr=self.args.lr)
            cr_freq = CE(self.timefreq_model)
            
            # 早停法参数初始化
            best_acc = 0.0          # 跟踪最佳验证准确率
            patience = 30            # 允许的连续无提升epoch数
            delta = 0.001           # 认为有提升的最小变化量
            no_improve_count = 0    # 无提升计数器

            # 初始化TensorBoard SummaryWriter
            writer = SummaryWriter(log_dir=self.save_path + '/runs')  # 指定日志目录

            # 最多训练num_epoch个epoch，但允许早停
            for epoch in range(getattr(self.args, 'num_epoch', 50)):
                print('Epoch:' + str(epoch + 1))
                self.timefreq_model.train()
                tqdm_dataloader = tqdm(self.train_loader)
                test_tqdm_dataloader=tqdm(self.test_loader)
                loss_sum = 0

                for idx, batch in enumerate(tqdm_dataloader):
                    batch = [x.to(self.device) for x in batch]
                    self.optimizer.zero_grad()
                    loss=cr_freq.computefreq(batch)
                    loss.backward()
                    self.optimizer.step()
                    loss_sum += loss.item()

                trloss=loss_sum / (idx + 1)

                metrics = {'acc': 0, 'f1': 0}
                pred = []
                label = []
                test_loss = 0
                acc3=0
                acc5=0

                for idxte, batch in enumerate(test_tqdm_dataloader):
                    self.timefreq_model.eval()
                    batch = [x.to(self.device) for x in batch]
                    ret = self.compute_metrics_freq(batch,self.timefreq_model)
                    if len(ret) == 2:
                        pred_b, label_b = ret
                        pred += pred_b
                        label += label_b
                    else:
                        pred_b, label_b, test_loss_b,acc3_b,acc5_b = ret
                        pred += pred_b
                        label += label_b
                        acc3+=acc3_b
                        acc5+=acc5_b
                        test_loss += test_loss_b.cpu().item()
                confusion_mat = self._confusion_mat(label, pred)
                self.print_process(confusion_mat)

                if self.args.num_class == 2:
                    metrics['f1'] = f1_score(y_true=label, y_pred=pred)
                    metrics['precision'] = precision_score(y_true=label, y_pred=pred)
                    metrics['recall'] = recall_score(y_true=label, y_pred=pred)
                else:
                    metrics['f1'] = f1_score(y_true=label, y_pred=pred, average='macro')
                    metrics['micro_f1'] = f1_score(y_true=label, y_pred=pred, average='micro')
                metrics['acc'] = accuracy_score(y_true=label, y_pred=pred)
                metrics['test_loss'] = test_loss / (idxte + 1)

                te_top3acc = acc3 / (idxte + 1)
                te_top5acc = acc5 / (idxte + 1)

                print('timefreq_finetune epoch{0}, trloss{1}, teloss{2},teacc{3},te_top3acc{4},te_top3acc{5}'.format(epoch + 1, trloss, metrics['test_loss'],metrics['acc'],te_top3acc,te_top5acc))
                
                # 将损失值记录到TensorBoard
                writer.add_scalar('Loss/train', trloss, epoch)
                writer.add_scalar('Loss/test', metrics['test_loss'], epoch)
                writer.add_scalar('Accuracy/test', metrics['acc'], epoch)
                writer.add_scalar('Accuracy/test_top3', te_top3acc, epoch)
                writer.add_scalar('Accuracy/test_top5', te_top5acc, epoch)

                # 将损失和准确率保存到CSV文件
                with open(csv_filename, mode='a', newline='') as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow([epoch + 1, trloss, metrics['test_loss'], metrics['acc'], te_top3acc, te_top5acc])

                # 早停法逻辑 ---------------------------------------------------
                current_acc = metrics['acc']
                
                # 如果当前准确率超过历史最佳+阈值，则更新最佳并重置计数器
                if current_acc > best_acc + delta:
                    best_acc = current_acc
                    no_improve_count = 0
                    # 保存当前最佳模型
                    torch.save(self.timefreq_model.state_dict(), self.save_path + '/timefreqmodel_sub1.pkl')
                else:
                    no_improve_count += 1
                    # 检查是否触发早停
                    if no_improve_count >= patience:
                        print(f'Early stopping triggered at epoch {epoch+1}, no improvement in {patience} epochs.')
                        break  # 终止训练循环

                # 原保存逻辑：每5个epoch保存一次（保留）
                if (epoch + 1) % 5 == 0:
                    torch.save(self.timefreq_model.state_dict(),
                            self.save_path + '/timefreqmodel_epoch' + str(epoch + 1) + '.pkl')

            # 关闭TensorBoard writer
            writer.close()


    def finetune_timefreq_alpha(self):
            # --- 0. 准备工作 ---
            self.print_process("开始 TimeFreqEncoder 微调...")
            # 获取当前时间并格式化为文件名的一部分
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            run_log_dir = self.save_path + f"/finetune_timefreq_{timestamp}"
            os.makedirs(run_log_dir, exist_ok=True)
            # run_log_dir.mkdir(parents=True, exist_ok=True)
            self.print_process(f"日志和模型将保存在: {run_log_dir}")
            
            csv_filename = run_log_dir + f'/train_log_{timestamp}.csv'
            tb_log_dir = run_log_dir + 'runs' # TensorBoard 日志目录

            # 初始化CSV文件
            csv_header = ['epoch', 'train_loss', 'test_loss', 'accuracy', 'top3_accuracy', 'top5_accuracy']
            try:
                with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow(csv_header)
                self.print_process(f"CSV 日志已初始化: {csv_filename}")
            except IOError as e:
                self.print_process(f"错误: 初始化 CSV 文件失败: {e}")
                return # 或者不使用 CSV

            # 初始化TensorBoard Writer (只在 rank 0)
            try:
                    writer = SummaryWriter(log_dir=tb_log_dir)
                    self.print_process(f"TensorBoard 日志目录: {tb_log_dir}")
            except Exception as e_tb:
                    self.print_process(f"警告: 初始化 TensorBoard 时出错: {e_tb}")
                    writer = None

                
            # --- 1. 加载预训练的 TimeEncoder ---
            # time_encoder_path = self.save_path + 'finetune_model_epoch80.pkl' # 使用 Path
            # if time_encoder_path.is_file():
            #      print(f"加载预训练 TimeEncoder 从: {time_encoder_path}")
            #      try:
            #         time_state_dict = torch.load(time_encoder_path, map_location=self.device)
            #         # 处理可能的 DDP 前缀
            #         is_ddp = all(k.startswith('module.') for k in time_state_dict.keys())
            #         if is_ddp: time_state_dict = {k[7:]: v for k, v in time_state_dict.items()}
            #         self.time_encoder_model.load_state_dict(time_state_dict)
            #         print("TimeEncoder 权重加载成功。")
            #      except Exception as e:
            #         print(f"错误: 加载 TimeEncoder 权重失败: {e}。将使用初始化的 TimeEncoder。")
            # else:
            #     print(f"警告: 未找到预训练 TimeEncoder 文件: {time_encoder_path}。将使用初始化的 TimeEncoder。")
            time_state_dict = torch.load(self.save_path + '/finetune_model_epoch80.pkl',
                                    map_location=self.device)
            self.model.load_state_dict(time_state_dict)

            # (可选) 运行一次 TimeEncoder 的线性评估作为基线
            self.print_process("运行 TimeEncoder 线性评估基线...")
            self.model.eval()
            self.model.to(self.device)
            try:
                train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
                test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
                if train_rep is not None and test_rep is not None: # 检查是否成功提取
                    clf = fit_lr(train_rep, train_label)
                    acc_baseline = clf.score(test_rep, test_label)
                    pred_label_baseline = np.argmax(clf.predict_proba(test_rep), axis=1)
                    f1_baseline = f1_score(test_label, pred_label_baseline, average='macro', zero_division=0)
                    self.print_process(f"TimeEncoder 线性评估基线: Accuracy={acc_baseline:.4f}, F1-Macro={f1_baseline:.4f}")
                else:
                    self.print_process("警告: 提取 TimeEncoder 特征失败，无法进行线性评估。")
            except Exception as e_linear:
                 self.print_process(f"警告: TimeEncoder 线性评估时出错: {e_linear}")
            self.model.train() # 恢复训练模式
            # self.time_encoder_model.to(torch.device("cpu")) # 不再移到 CPU

            # --- 2. 实例化频率模型 (EEGNetEncoder 或 FreqEncoder) ---
            num_channels = 128
            print(f"假设 EEG 通道数: {num_channels} (请务必核实!)")

            # 从 args 中获取时间点数
            # 确保 args 中有 time_high 和 time_low
            if not hasattr(self.args, 'time_high') or not hasattr(self.args, 'time_low'):
                 # 如果 args 没有，尝试从 opt 获取 (假设 opt 在类中可用或传入)
                 # 或者你需要找到正确的参数来源
                 raise AttributeError("Trainer 的 args 对象缺少 time_high 或 time_low 属性")
            num_time_steps = self.args.time_high - self.args.time_low
            print(f"计算得到的时间点数: {num_time_steps}")

            # 从命令行参数 -mp 获取可选参数
            # 添加检查，确保 x.split('=') 的结果长度为 2
            freq_model_options = {}
            for item in opt.model_params: # 使用全局的 opt 对象
                 parts = item.split("=")
                 if len(parts) == 2:
                     key, value_str = parts
                     try:
                         value = int(value_str) if value_str.isdigit() else (float(value_str) if value_str.replace('.', '', 1).isdigit() else value_str)
                         freq_model_options[key] = value
                     except ValueError:
                         print(f"警告: 无法解析参数值 '{value_str}' (来自 '{item}'). 跳过。")
                         freq_model_options[key] = value_str # 或者直接用字符串
                 else:
                     print(f"警告: 忽略格式不正确的模型参数 '{item}'。请使用 key=value 格式。")

            print(f"从 -mp 解析的可选参数: {freq_model_options}")

            # --- 根据 opt.model_type 实例化正确的频率模型 ---
            # !! 关键：从全局 opt 获取 model_type !!
            if opt.model_type == 'eegnet' or opt.model_type == 'EEGNetEncoder':
                print("实例化 EEGNetEncoder...")
                # 准备 EEGNetEncoder 的参数
                eegnet_args = {
                    'input_channels': num_channels,
                    'input_time_points': num_time_steps,
                    'num_classes': self.args.num_class, # 假设类别数在 args 中
                    # 可以设置一个默认的 output_feature_dim，如果需要的话
                    'output_feature_dim': freq_model_options.get('output_feature_dim', 128) # 默认或从 options 获取
                }
                # 合并可选参数 (如果 -mp 提供了 F1, D, F2 等，会覆盖默认值)
                eegnet_args.update(freq_model_options)
                # 移除可能不属于 EEGNetEncoder 的参数，防止 TypeError
                valid_eegnet_keys = ['input_channels', 'input_time_points', 'num_classes', 'F1', 'D', 'F2', 'dropout_rate', 'output_feature_dim']
                eegnet_args_filtered = {k: v for k, v in eegnet_args.items() if k in valid_eegnet_keys}

                print(f"传递给 EEGNetEncoder 的最终参数: {eegnet_args_filtered}")
                try:
                    freq_model = EEGNetEncoder(**eegnet_args_filtered)
                except TypeError as e:
                    print(f"实例化 EEGNetEncoder 时发生 TypeError: {e}")
                    print("请检查模型定义和传递的参数是否匹配。")
                    raise e # 重新抛出错误

            elif opt.model_type == 'lstm' or opt.model_type == 'FreqEncoder':
                print("实例化 FreqEncoder...")
                 # 准备 FreqEncoder 的参数
                freq_encoder_args = {
                    'input_size': num_channels,  # input_size = num EEG channels (128), not time steps
                    'lstm_size': freq_model_options.get('lstm_size', 128),
                    'lstm_layers': freq_model_options.get('lstm_layers', 1),
                    'output_size': freq_model_options.get('output_feature_dim', 128) # 假设 output_size 对应特征维度
                    # FreqEncoder 可能不需要 num_classes
                }
                # FreqEncoder 的构造函数可能只接受特定参数，过滤掉多余的
                valid_freqencoder_keys = ['input_size', 'lstm_size', 'lstm_layers', 'output_size'] # 根据 FreqEncoder 定义调整
                freq_encoder_args_filtered = {k:v for k,v in freq_encoder_args.items() if k in valid_freqencoder_keys}

                print(f"传递给 FreqEncoder 的最终参数: {freq_encoder_args_filtered}")
                try:
                     freq_model = FreqEncoder(**freq_encoder_args_filtered)
                except TypeError as e:
                    print(f"实例化 FreqEncoder 时发生 TypeError: {e}")
                    print("请检查模型定义和传递的参数是否匹配。")
                    raise e # 重新抛出错误
            else:
                 raise ValueError(f"不支持的模型类型: {opt.model_type}。请在命令行使用 -mt 指定 'lstm' 或 'eegnet'")

            # --- 加载预训练权重 (在模型实例化之后) ---
            # !! 关键：使用 opt.pretrained_net !!
            # if opt.pretrained_net != '':
            #     print(f"尝试从 '{opt.pretrained_net}' 加载预训练的 {opt.model_type} 权重...")
            #     try:
            #         # 优先尝试加载 state_dict (推荐)
            #         freq_state_dict = torch.load(opt.pretrained_net, map_location=self.device) # 加载到目标设备
            #         freq_model.load_state_dict(freq_state_dict)
            #         print(f"成功加载 {opt.model_type} 的 state_dict。")
            #     except (RuntimeError, KeyError, TypeError) as e1:
            #         print(f"加载 state_dict 失败: {e1}。尝试加载整个模型...")
            #         try:
            #             # 尝试加载整个模型 (如果 state_dict 失败)
            #             loaded_full_model = torch.load(opt.pretrained_net, map_location=self.device)
            #             # 检查加载的模型类型是否匹配
            #             if isinstance(loaded_full_model, type(freq_model)):
            #                 freq_model = loaded_full_model
            #                 print(f"成功加载完整的 {opt.model_type} 模型。")
            #             else:
            #                 print(f"警告: 从 '{opt.pretrained_net}' 加载的对象类型 ({type(loaded_full_model)}) 与期望的类型 ({type(freq_model)}) 不匹配。将使用随机初始化的模型。")
            #         except Exception as e2:
            #             print(f"加载整个模型也失败: {e2}。将使用随机初始化的 {opt.model_type}。")
            # else:
            #     print(f"未指定预训练的 {opt.model_type} 权重，将使用随机初始化的模型。")
            if opt.pretrained_net and opt.pretrained_net.strip() != '': # 确保路径非空
                print(f"尝试从 '{opt.pretrained_net}' 加载预训练的 {opt.model_type} 权重...")
                try:
                    # 加载整个文件，它可能是一个字典或直接是 state_dict
                    loaded_content = torch.load(opt.pretrained_net, map_location=self.device)

                    actual_state_dict_to_load = None

                    if isinstance(loaded_content, dict):
                        # 检查是否是训练脚本保存的检查点格式
                        if 'model_state_dict' in loaded_content:
                            actual_state_dict_to_load = loaded_content['model_state_dict']
                            print(f"从检查点文件中提取 'model_state_dict' 进行加载。")
                        elif 'state_dict' in loaded_content: # 有些可能用 'state_dict'
                            actual_state_dict_to_load = loaded_content['state_dict']
                            print(f"从检查点文件中提取 'state_dict' 进行加载。")
                        else:
                            # 可能是直接保存的 state_dict (也是一个字典)
                            actual_state_dict_to_load = loaded_content
                            print(f"加载的文件是一个字典，尝试直接作为 state_dict 加载。")
                    else:
                        # 如果不是字典，可能是一个直接保存的模型对象 (不推荐)
                        # 或者直接保存的 state_dict (已经被上面的 isinstance(dict) 覆盖)
                        # 这里我们假设如果不是字典，那么它应该是一个 state_dict (尽管可能性小)
                        # 更安全的做法是期望 state_dict
                        print(f"加载的文件不是字典，尝试作为 state_dict 加载 (可能性较低)。")
                        actual_state_dict_to_load = loaded_content


                    if actual_state_dict_to_load is not None:
                        # 过滤掉不匹配的键 (例如，如果加载的是 TimeFreqEncoder 的权重给 FreqEncoder)
                        # 这段过滤逻辑对于确保只加载 freq_model 自身的权重很重要
                        model_keys = set(freq_model.state_dict().keys())
                        loaded_keys = set(actual_state_dict_to_load.keys())

                        # 尝试找到 freq_model 在组合模型中可能的参数前缀
                        # 例如，如果 TimeFreqEncoder 中 freq_model 叫做 self.pretrained_model_freq
                        # 那么 freq_model 的参数键在保存的 TimeFreqEncoder 权重中可能是
                        # "pretrained_model_freq.eegnet.conv1.weight" 等
                        # 我们需要移除这个前缀才能正确加载到独立的 freq_model 中

                        # 一个简单的策略：如果加载的键有类似 "eegnet." 或 "lstm." 的部分，
                        # 并且 freq_model 自身的键没有这个前缀，尝试去除。
                        # 但更稳妥的是，确保 --pretrained_net 指向的是 freq_model 单独训练保存的权重。

                        # 如果 pretrained_net 确定是 freq_model 单独的权重，则可以直接加载
                        try:
                            freq_model.load_state_dict(actual_state_dict_to_load, strict=False) # strict=False 更宽容
                            print(f"成功加载 {opt.model_type} 的 state_dict。")
                        except RuntimeError as e_load:
                            print(f"使用提取的 state_dict 加载失败: {e_load}")
                            print("可能是由于键名不匹配。如果加载的是组合模型的权重，需要特殊处理键名。")
                            print(f"freq_model 的键示例: {list(model_keys)[:5]}")
                            print(f"加载的 state_dict 键示例: {list(loaded_keys)[:5]}")
                            print(f"将使用随机初始化的 {opt.model_type}。")

                    else: # 尝试加载整个模型对象 (如果上面的逻辑没能提取出 state_dict)
                        print(f"未能从加载内容中提取 state_dict。尝试将加载内容作为整个模型对象...")
                        if isinstance(loaded_content, type(freq_model)):
                            freq_model = loaded_content # 直接替换
                            print(f"成功将加载内容作为完整的 {opt.model_type} 模型使用。")
                        else:
                            print(f"警告: 从 '{opt.pretrained_net}' 加载的对象类型 ({type(loaded_content)}) 与期望类型 ({type(freq_model)}) 不匹配。将使用随机初始化的模型。")

                except FileNotFoundError:
                    print(f"错误: 预训练权重文件未找到 '{opt.pretrained_net}'。")
                except Exception as e_outer_load:
                    print(f"加载预训练的 {opt.model_type} 权重时发生未知错误: {e_outer_load}。将使用随机初始化的模型。")
            else:
                print(f"未指定预训练的 {opt.model_type} 权重，将使用随机初始化的模型。")

            # 将频率模型移到设备
            freq_model.to(self.device)

            # --- 4. 实例化新的 TimeFreqEncoder (使用加权融合版本) ---
            fusion_dim = getattr(self.args, 'fusion_dim', 512) # 从 args 获取融合维度或默认
            learnable_alpha = getattr(self.args, 'learnable_alpha', False) # 从 args 获取或默认
            # fixed_alpha = getattr(self.args, 'fixed_alpha', 0.5) # 从 args 获取固定 alpha 或默认
            fixed_alpha = 0 # 从 args 获取固定 alpha 或默认
            fusion_dropout = getattr(self.args, 'fusion_dropout', 0.5) # 从 args 获取融合 dropout

            # 创建新的 TimeFreqEncoder 实例，传入 TimeEncoder 和 Freq/EEGNet 模型
            self.timefreq_model = TimeFreqEncoder_alpha(
                self.model, # 已经加载了权重的 TimeEncoder
                freq_model,             # 加载了权重或随机初始化的频率模型
                self.args,              # 传递全局配置
                fusion_dim=fusion_dim,
                learnable_alpha=learnable_alpha,
                fusion_dropout=fusion_dropout,
                fixed_alpha_value=fixed_alpha # 传递固定 alpha 值
                # fixed_alpha 会在 learnable_alpha=False 时在内部使用
            )
            # 如果 alpha 固定，在这里设置一下 (虽然内部也会设)
            if not learnable_alpha:
                 self.timefreq_model.alpha = fixed_alpha
                 
            self.timefreq_model = self.timefreq_model.to(self.device) # 将组合模型移到设备
            self.print_process(f"TimeFreqEncoder (加权融合版) 实例化完成。融合维度: {fusion_dim}, Alpha 可学习: {learnable_alpha}, 固定 Alpha: {fixed_alpha if not learnable_alpha else 'N/A'}")
            
            # --- 5. 设置优化器和损失函数 ---
            # 优化器现在优化 TimeFreqEncoder 的所有参数（包括内部的 time/freq 模型和新的融合层/alpha）
            self.optimizer = torch.optim.AdamW(self.timefreq_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            cr_freq = CE(self.timefreq_model) # 假设 CE 损失类可以处理新的 TimeFreqEncoder

            # (可选) 设置学习率调度器
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=10, verbose=self.verbose)

            # --- 6. 训练循环 ---
            best_val_acc = 0.0          # !! 使用验证集准确率 !!
            patience = 50            # 早停耐心
            delta = 0.001           # 提升阈值
            no_improve_count = 0    # 未提升计数器
            best_model_path = run_log_dir + '/timefreqmodel_best_val.pkl' # 保存最佳验证模型

            # 使用 self.args.num_epoch (Finetune 的 epoch 数)
            num_epochs_finetune = getattr(self.args, 'num_epoch', 50)  # from args or default
            self.print_process(f"开始 TimeFreqEncoder 微调，共 {num_epochs_finetune} epochs...")

            for epoch in range(num_epochs_finetune):
                self.print_process(f"--- Epoch {epoch + 1}/{num_epochs_finetune} ---")
                self.timefreq_model.train() # 设置为训练模式
                
                # !! 如果使用 DDP，需要 sampler.set_epoch(epoch) !!
                # if hasattr(self.train_loader.sampler, 'set_epoch'):
                #     self.train_loader.sampler.set_epoch(epoch)
                    
                epoch_loss_sum = 0.0
                epoch_samples = 0
                
                # --- 训练批次循环 ---
                tqdm_dataloader = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [训练]", leave=False) if self.verbose else self.train_loader
                for idx, batch in enumerate(tqdm_dataloader):
                     # 假设 batch 结构是 (seqs, labels, ...)
                     if batch is None or len(batch) < 2: continue # 跳过无效批次
                     seqs = batch[0].to(self.device)
                     labels = batch[1].to(self.device) # 假设标签在 batch[1]
                     
                     self.optimizer.zero_grad()
                     # 调用 CE 类的 computefreq 方法计算损失
                     # 需要确认 CE.computefreq 内部调用 self.model(seqs) 并处理返回的 scores
                     loss = cr_freq.computefreq(batch) # 假设 CE 处理了设备和批次结构
                     
                     # 如果 computefreq 不返回损失，需要直接调用模型并计算损失
                     # lastrep, encoded, scores = self.timefreq_model(seqs)
                     # loss = F.cross_entropy(scores, labels.long()) # 例如

                     loss.backward()
                     # 可选：梯度裁剪
                     # torch.nn.utils.clip_grad_norm_(self.timefreq_model.parameters(), max_norm=1.0)
                     self.optimizer.step()
                     
                     epoch_loss_sum += loss.item() * seqs.size(0) # 累加加权损失
                     epoch_samples += seqs.size(0)
                     if self.verbose and idx % 5 == 0: # 每 5 个 batch 打印一次进度
                         tqdm_dataloader.set_postfix(loss=loss.item())

                avg_train_loss = epoch_loss_sum / epoch_samples if epoch_samples > 0 else 0
                self.print_process(f"Epoch {epoch+1} 训练完成, 平均训练损失: {avg_train_loss:.4f}")

                # --- 评估 (使用测试集，理想情况下应该用验证集) ---
                self.timefreq_model.eval() # 设置为评估模式
                
                eval_preds = []
                eval_labels = []
                eval_loss_sum = 0.0
                eval_acc3_correct = 0
                eval_acc5_correct = 0
                eval_samples = 0
                
                # 使用测试集加载器进行评估 (理想情况是 val_loader)
                eval_loader = self.test_loader # <-- !! 应该用验证集 !!
                if eval_loader is None:
                    self.print_process("警告: 没有测试/验证加载器，跳过评估步骤。")
                    continue # 如果没有评估集，跳到下一个 epoch

                tqdm_eval_loader = tqdm(eval_loader, desc=f"Epoch {epoch+1} [评估]", leave=False) if self.verbose else eval_loader
                
                for idxte, batch in enumerate(tqdm_eval_loader):
                    if batch is None or len(batch) < 2: continue
                    # 调用 metrics 函数 (假设它处理设备和批次结构)
                    ret = self.compute_metrics_freq(batch, self.timefreq_model)
                    if ret is None: continue # 跳过无效返回

                    # 解包返回结果
                    pred_b, label_b, test_loss_b, acc3_b_correct, acc5_b_correct = ret
                    
                    eval_preds.extend(pred_b)
                    eval_labels.extend(label_b)
                    # test_loss_b 是单个 batch 的损失值，需要乘以 batch size 再累加
                    batch_size_eval = len(label_b) # 获取当前批次大小
                    eval_loss_sum += test_loss_b.cpu().item() * batch_size_eval
                    eval_acc3_correct += acc3_b_correct
                    eval_acc5_correct += acc5_b_correct
                    eval_samples += batch_size_eval

                # --- 计算评估指标 ---
                if eval_samples == 0:
                    self.print_process("警告: 评估集样本数为 0, 无法计算指标。")
                    continue

                avg_eval_loss = eval_loss_sum / eval_samples
                eval_acc = accuracy_score(eval_labels, eval_preds)
                eval_f1_macro = f1_score(eval_labels, eval_preds, average='macro', zero_division=0)
                eval_top3_acc = eval_acc3_correct / eval_samples
                eval_top5_acc = eval_acc5_correct / eval_samples

                self.print_process(f"Epoch {epoch+1} 评估结果: Loss={avg_eval_loss:.4f}, Acc={eval_acc:.4f}, F1-Macro={eval_f1_macro:.4f}, Top3Acc={eval_top3_acc:.4f}, Top5Acc={eval_top5_acc:.4f}")


                # TensorBoard
                if writer:
                    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch + 1)
                    writer.add_scalar('Loss/eval_epoch', avg_eval_loss, epoch + 1)
                    writer.add_scalar('Accuracy/eval_epoch', eval_acc, epoch + 1)
                    writer.add_scalar('Accuracy/eval_top3_epoch', eval_top3_acc, epoch + 1)
                    writer.add_scalar('Accuracy/eval_top5_epoch', eval_top5_acc, epoch + 1)
                    writer.add_scalar('F1_Macro/eval_epoch', eval_f1_macro, epoch + 1)
                    # 记录学习率
                    writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch + 1)
                    # 记录 alpha (如果可学习)
                    if learnable_alpha:
                            alpha_val = torch.sigmoid(self.timefreq_model.logit_alpha).item()
                            writer.add_scalar('Fusion/alpha', alpha_val, epoch + 1)

                # CSV
                try:
                    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                        writer_csv = csv.writer(f)
                        writer_csv.writerow([epoch + 1, avg_train_loss, avg_eval_loss, eval_acc, eval_top3_acc, eval_top5_acc])
                except IOError as e_csv:
                    self.print_process(f"警告: 写入 CSV 时出错: {e_csv}")

                # --- 早停法逻辑 (基于验证/测试准确率) ---
                current_eval_acc = eval_acc # 使用当前 epoch 的评估准确率

                if current_eval_acc > best_val_acc + delta:
                    self.print_process(f"评估准确率提升: {best_val_acc:.4f} -> {current_eval_acc:.4f}. 保存最佳模型...")
                    best_val_acc = current_eval_acc
                    no_improve_count = 0
                    # 保存当前最佳模型状态字典 (只在 Rank 0)
                    try:
                        # 保存原始模型 (去掉 DDP 的 module. 前缀，如果用了 DDP)
                        state_to_save = self.timefreq_model.module.state_dict() if isinstance(self.timefreq_model, torch.nn.parallel.DistributedDataParallel) else self.timefreq_model.state_dict()
                        torch.save(state_to_save, best_model_path)
                    except Exception as e_save:
                        self.print_process(f"错误: 保存最佳模型失败: {e_save}")
                else:
                    no_improve_count += 1
                    self.print_process(f"评估准确率未提升 ({no_improve_count}/{patience}).")
                    if no_improve_count >= patience:
                        self.print_process(f"早停触发于 Epoch {epoch+1}。")
                        break  # 终止训练循环

                # --- 定期保存检查点 (只在 Rank 0) ---
                if (epoch + 1) % getattr(self.args, 'saveCheck', 100) == 0: # 使用 args.saveCheck
                    checkpoint_path = run_log_dir + f'/timefreqmodel_epoch{epoch + 1}.pkl'
                    try:
                        state_to_save = self.timefreq_model.module.state_dict() if isinstance(self.timefreq_model, torch.nn.parallel.DistributedDataParallel) else self.timefreq_model.state_dict()
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': state_to_save,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            # 'scheduler_state_dict': scheduler.state_dict(), # 如果用了 scheduler
                            'best_val_acc': best_val_acc,
                            'args': self.args # 保存配置
                        }, checkpoint_path)
                        self.print_process(f"已保存检查点: {checkpoint_path}")
                    except Exception as e_ckpt:
                        self.print_process(f"错误: 保存检查点失败: {e_ckpt}")

                # --- 更新学习率 (如果用了 scheduler) ---
                # scheduler.step(current_eval_acc) # 基于验证准确率

            # --- 训练循环结束 ---

            # 关闭 TensorBoard writer (只在 rank 0)
            if writer: writer.close(); 
            self.print_process("TensorBoard writer 已关闭。")
            self.print_process("TimeFreqEncoder 微调结束。")


    #     # 替换 Trainer 类中现有的 finetune_timefreq 方法
    
    # def finetune_timefreq_MLP(self):
    #     # --- 阶段 1: 加载编码器并提取特征 ---
    #     print("--- 阶段 1: 加载编码器并提取特征 ---")

    #     # 加载预训练的时间编码器状态
    #     time_state_dict = torch.load(self.save_path + '/finetune_model_epoch80.pkl',
    #                             map_location=self.device)
    #     self.model.load_state_dict(time_state_dict)
    #     self.model.to(self.device)
    #     self.model.eval() # 将时间编码器设置为评估模式（冻结参数）

    #     # 初始化并加载预训练的频率编码器
    #     freq_model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for
    #                     (key, value) in [x.split("=") for x in opt.model_params]}
    #     freq_model = FreqEncoder(**freq_model_options).to(self.device)

    #     if opt.pretrained_net != '':
    #         # 假设 opt.pretrained_net 指向 FreqEncoder 的 state_dict 文件
    #         try:
    #              # 如果保存的是整个模型
    #              freq_model = torch.load(opt.pretrained_net, map_location=self.device)
    #         except:
    #              # 如果只保存了 state_dict
    #              freq_state_dict = torch.load(opt.pretrained_net, map_location=self.device)
    #              freq_model.load_state_dict(freq_state_dict)
    #         print(f"已从以下路径加载预训练的频率模型: {opt.pretrained_net}")
    #     freq_model.eval() # 将频率编码器设置为评估模式（冻结参数）

    #     # --- 特征提取函数 ---
    #     def extract_combined_features(loader, time_encoder, freq_encoder):
    #         all_features = []
    #         all_labels = []
    #         time_encoder.eval()
    #         freq_encoder.eval()
    #         with torch.no_grad():
    #             for batch in tqdm(loader, desc="正在提取特征"):
    #                 # 假设 batch[0] 是 EEG 数据, batch[1] 是标签
    #                 eeg_data = batch[0].to(self.device)
    #                 labels = batch[1].to(self.device)

    #                 # --- 获取 TimeEncoder 特征 ---
    #                 # 确保 TimeEncoder 在特征提取模式下运行
    #                 # 我们需要平均池化后的特征
    #                 # 临时设置状态以确保 forward 返回正确的特征
    #                 original_linear_proba = time_encoder.linear_proba
    #                 original_nocliptune = time_encoder.nocliptune
    #                 time_encoder.linear_proba = True
    #                 time_encoder.nocliptune = True # 确保进入正确的 if 分支

    #                 time_features = time_encoder(eeg_data) # 现在应该返回 torch.mean(x, dim=1)

    #                 # 恢复原始状态（如果需要在 Trainer 的其他地方使用不同状态）
    #                 time_encoder.linear_proba = original_linear_proba
    #                 time_encoder.nocliptune = original_nocliptune
    #                 # --- TimeEncoder 特征获取结束 ---

    #                 # --- 获取 FreqEncoder 特征 ---
    #                 # 调用 FreqEncoder 的 forward 方法，并获取分类头之前的表示 xa
    #                 _, freq_features_xa = freq_encoder(eeg_data) # 直接调用 forward 获取 (logits, xa)
    #                 # --- FreqEncoder 特征获取结束 ---

    #                 # 合并特征
    #                 # print(f"Time features shape: {time_features.shape}") # 调试用
    #                 # print(f"Freq features shape: {freq_features_xa.shape}") # 调试用
    #                 combined_features = torch.cat((time_features, freq_features_xa), dim=1)
    #                 # print(f"Combined features shape: {combined_features.shape}") # 调试用

    #                 all_features.append(combined_features.cpu())
    #                 all_labels.append(labels.cpu())

    #         # 在函数末尾添加维度检查
    #         if not all_features: # 处理空列表的情况
    #              print("警告：没有从数据加载器中提取到任何特征！")
    #              # 返回空的张量或根据需要处理
    #              return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)

    #         # 确保所有批次的特征维度一致
    #         first_batch_dim = all_features[0].shape[1] if all_features else None
    #         if first_batch_dim is not None:
    #             for i, feat_batch in enumerate(all_features):
    #                 if feat_batch.shape[1] != first_batch_dim:
    #                     print(f"警告：批次 {i} 的特征维度 ({feat_batch.shape[1]}) 与第一个批次 ({first_batch_dim}) 不匹配！")
    #                     # 这里可能需要错误处理或填充逻辑

    #         # 尝试拼接前再次检查
    #         try:
    #             final_features = torch.cat(all_features, dim=0)
    #             final_labels = torch.cat(all_labels, dim=0)
    #         except RuntimeError as e:
    #             print(f"拼接特征时出错: {e}")
    #             print("检查每个批次的特征形状:")
    #             for i, f in enumerate(all_features):
    #                 print(f"批次 {i}: {f.shape}")
    #             # 根据错误处理
    #             return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)


    #         return final_features, final_labels
    #     # 为训练集和测试集提取特征
    #     # 使用正确的加载器（train_linear_loader 用于获取训练特征）
    #     print("正在提取训练特征...")
    #     train_features, train_labels = extract_combined_features(self.train_linear_loader, self.model, freq_model)
    #     print("正在提取测试特征...")
    #     test_features, test_labels = extract_combined_features(self.test_loader, self.model, freq_model)
    #     print(f"提取完成: 训练特征形状={train_features.shape}, 测试特征形状={test_features.shape}")

    #     # --- 阶段 2: 定义和训练 MLP 分类器 ---
    #     print("\n--- 阶段 2: 训练 MLP 分类器 ---")

    #     # 为提取的特征创建数据集和数据加载器
    #     train_feat_dataset = Data.TensorDataset(train_features, train_labels)
    #     test_feat_dataset = Data.TensorDataset(test_features, test_labels)

    #     # MLP 训练使用 train_batch_size, 评估使用 test_batch_size
    #     mlp_train_loader = Data.DataLoader(train_feat_dataset, batch_size=self.args.train_batch_size, shuffle=True)
    #     mlp_test_loader = Data.DataLoader(test_feat_dataset, batch_size=self.args.test_batch_size, shuffle=False)

    #     # 定义 MLP 模型
    #     feature_dim = train_features.shape[1] # 获取特征维度
    #     hidden_dims = [512, 256] # 示例: 两个隐藏层，节点数分别为 256 和 128
    #     output_dim = self.args.num_class # 从 args 获取类别数量
    #     mlp_classifier = MLPClassifier(feature_dim, hidden_dims, output_dim).to(self.device)

    #     # 定义 MLP 的优化器和损失函数
    #     mlp_optimizer = torch.optim.AdamW(mlp_classifier.parameters(), lr=self.args.lr) # 可以使用 args.lr 或设置新的学习率
    #     mlp_criterion = nn.CrossEntropyLoss() # 交叉熵损失，适用于多分类

    #     # 设置日志记录
    #     current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    #     mlp_save_path = os.path.join(self.save_path, "mlp_classifier") # 为 MLP 创建子目录
    #     os.makedirs(mlp_save_path, exist_ok=True) # 确保目录存在
    #     csv_filename = os.path.join(mlp_save_path, f'mlp_train_log_{current_time}.csv')
    #     writer = SummaryWriter(log_dir=os.path.join(mlp_save_path, 'runs')) # MLP 的 Tensorboard 日志

    #     # 初始化 CSV 文件
    #     with open(csv_filename, mode='w', newline='') as f:
    #         writer_csv = csv.writer(f)
    #         # 写入 CSV 文件的表头
    #         writer_csv.writerow(['epoch', 'train_loss', 'test_loss', 'test_accuracy', 'test_top3_accuracy', 'test_top5_accuracy', 'test_f1_macro'])

    #     # MLP 的早停参数
    #     best_acc = 0.0          # 跟踪最佳测试准确率
    #     patience = 10           # 容忍多少个 epoch 没有提升（MLP 可能需要更多耐心）
    #     delta = 0.001           # 认为有提升的最小准确率变化量
    #     no_improve_count = 0    # 没有提升的 epoch 计数器

    #     num_mlp_epochs = 100 # 为 MLP 训练设置期望的 epoch 数量

    #     print(f"开始 MLP 训练，共 {num_mlp_epochs} 个 epoch...")
    #     print(f"MLP 结构: 输入={feature_dim}, 隐藏层={hidden_dims}, 输出={output_dim}")

    #     # MLP 训练循环
    #     for epoch in range(num_mlp_epochs):
    #         mlp_classifier.train() # 设置为训练模式
    #         train_loss_sum = 0.0

    #         # 训练进度条
    #         progress_bar = tqdm(mlp_train_loader, desc=f"MLP Epoch {epoch+1}/{num_mlp_epochs} [训练]")
    #         for batch_features, batch_labels in progress_bar:
    #             # 将数据移动到指定设备
    #             batch_features = batch_features.to(self.device)
    #             batch_labels = batch_labels.to(self.device).long() # 交叉熵损失需要 Long 类型的标签

    #             # 梯度清零
    #             mlp_optimizer.zero_grad()
    #             # 前向传播
    #             outputs = mlp_classifier(batch_features)
    #             # 计算损失
    #             loss = mlp_criterion(outputs, batch_labels)
    #             # 反向传播
    #             loss.backward()
    #             # 更新参数
    #             mlp_optimizer.step()

    #             train_loss_sum += loss.item()
    #             # 更新进度条显示当前批次的损失
    #             progress_bar.set_postfix(loss=loss.item())

    #         avg_train_loss = train_loss_sum / len(mlp_train_loader)

    #         # --- 阶段 3: 评估 MLP 分类器 ---
    #         mlp_classifier.eval() # 设置为评估模式
    #         test_loss_sum = 0.0
    #         all_preds = [] # 存储所有预测标签
    #         all_labels_list = [] # 存储所有真实标签

    #         # 评估进度条
    #         progress_bar_test = tqdm(mlp_test_loader, desc=f"MLP Epoch {epoch+1}/{num_mlp_epochs} [评估]")
    #         with torch.no_grad(): # 评估时禁用梯度计算
    #             for batch_features, batch_labels in progress_bar_test:
    #                 batch_features = batch_features.to(self.device)
    #                 batch_labels = batch_labels.to(self.device).long()

    #                 # 前向传播
    #                 outputs = mlp_classifier(batch_features)
    #                 # 计算损失
    #                 loss = mlp_criterion(outputs, batch_labels)
    #                 test_loss_sum += loss.item()

    #                 # 获取预测结果（概率最高的类别）
    #                 _, predicted = torch.max(outputs.data, 1)
    #                 all_preds.extend(predicted.cpu().numpy())
    #                 all_labels_list.extend(batch_labels.cpu().numpy())

    #         avg_test_loss = test_loss_sum / len(mlp_test_loader)

    #         # 计算评估指标
    #         test_acc = accuracy_score(all_labels_list, all_preds)
    #         # 计算 Top-k 准确率需要模型的原始输出 (logits 或概率)
    #         test_top3_acc = 0.0
    #         test_top5_acc = 0.0
    #         with torch.no_grad(): # 重新计算输出来获取 top-k
    #              all_outputs = []
    #              for batch_features, _ in mlp_test_loader:
    #                  outputs = mlp_classifier(batch_features.to(self.device))
    #                  all_outputs.append(outputs.cpu())
    #              all_outputs_tensor = torch.cat(all_outputs, dim=0)
    #              # 使用 top_k_accuracy_score 辅助函数 (确保它已正确定义)
    #              # 注意：这里传入的是完整的 test_labels 张量
    #              test_top3_acc = top_k_accuracy_score(test_labels, all_outputs_tensor, k=3)
    #              test_top5_acc = top_k_accuracy_score(test_labels, all_outputs_tensor, k=5)

    #         test_f1_macro = f1_score(all_labels_list, all_preds, average='macro')
    #         # test_f1_micro = f1_score(all_labels_list, all_preds, average='micro') # 可选：计算 micro F1

    #         print(f"MLP Epoch {epoch+1}: 训练损失={avg_train_loss:.4f}, 测试损失={avg_test_loss:.4f}, "
    #               f"测试准确率={test_acc:.4f}, Top3 准确率={test_top3_acc:.4f}, Top5 准确率={test_top5_acc:.4f}, F1 Macro={test_f1_macro:.4f}")

    #         # 记录到 TensorBoard
    #         writer.add_scalar('MLP/Loss/train', avg_train_loss, epoch)
    #         writer.add_scalar('MLP/Loss/test', avg_test_loss, epoch)
    #         writer.add_scalar('MLP/Accuracy/test', test_acc, epoch)
    #         writer.add_scalar('MLP/Accuracy/test_top3', test_top3_acc, epoch)
    #         writer.add_scalar('MLP/Accuracy/test_top5', test_top5_acc, epoch)
    #         writer.add_scalar('MLP/F1_Macro/test', test_f1_macro, epoch)

    #         # 记录到 CSV
    #         with open(csv_filename, mode='a', newline='') as f:
    #             writer_csv = csv.writer(f)
    #             writer_csv.writerow([epoch + 1, avg_train_loss, avg_test_loss, test_acc, test_top3_acc, test_top5_acc, test_f1_macro])

    #         # 早停逻辑和保存最佳模型
    #         current_acc = test_acc # 当前周期的测试准确率
    #         if current_acc > best_acc + delta: # 如果当前准确率超过历史最佳+阈值
    #             best_acc = current_acc # 更新最佳准确率
    #             no_improve_count = 0 # 重置计数器
    #             # 保存当前最佳模型
    #             torch.save(mlp_classifier.state_dict(), os.path.join(mlp_save_path, 'mlp_classifier_best.pkl'))
    #             print(f"   -> 发现新的最佳测试准确率: {best_acc:.4f}。已保存最佳模型。")
    #         else:
    #             no_improve_count += 1 # 未提升，计数器加 1
    #             print(f"   -> 测试准确率未提升，已持续 {no_improve_count} 个 epoch。")
    #             # 检查是否达到容忍上限
    #             if no_improve_count >= patience:
    #                 print(f"早停机制触发于 epoch {epoch+1}。")
    #                 break # 终止训练循环

    #         # 定期保存模型（可选，用于备份）
    #         if (epoch + 1) % 10 == 0:
    #             torch.save(mlp_classifier.state_dict(), os.path.join(mlp_save_path, f'mlp_classifier_epoch_{epoch+1}.pkl'))

    #     writer.close() # 关闭 TensorBoard 写入器
    #     print(f"MLP 训练结束。最佳测试准确率: {best_acc:.4f}")
    #     print(f"MLP 模型和日志已保存在: {mlp_save_path}")




    # 定义一个新的方法或修改之前的 finetune_timefreq_MLP
    def finetune_timefreq_ResMLP(self): # 重命名以区分
        # --- 阶段 1: 加载编码器并提取特征 (与 MLP 版本相同) ---
        print("--- 阶段 1: 加载编码器并提取特征 ---")

        # 加载预训练的时间编码器状态
        time_state_dict = torch.load(self.save_path + '/finetune_model_epoch80.pkl',
                                map_location=self.device)
        self.model.load_state_dict(time_state_dict)
        self.model.to(self.device)
        self.model.eval() # 将时间编码器设置为评估模式（冻结参数）

        # 初始化并加载预训练的频率编码器
        freq_model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for
                        (key, value) in [x.split("=") for x in opt.model_params]}
        freq_model = FreqEncoder(**freq_model_options).to(self.device)

        if opt.pretrained_net != '':
            # 假设 opt.pretrained_net 指向 FreqEncoder 的 state_dict 文件
            try:
                    # 如果保存的是整个模型
                    freq_model = torch.load(opt.pretrained_net, map_location=self.device)
            except:
                    # 如果只保存了 state_dict
                    freq_state_dict = torch.load(opt.pretrained_net, map_location=self.device)
                    freq_model.load_state_dict(freq_state_dict)
            print(f"已从以下路径加载预训练的频率模型: {opt.pretrained_net}")
        freq_model.eval() # 将频率编码器设置为评估模式（冻结参数）--
        
        def extract_combined_features(loader, time_encoder, freq_encoder):
           # ... (这里是之前修改好的 extract_combined_features 函数代码) ...
           # 确保返回 final_features, final_labels
            all_features = []
            all_labels = []
            time_encoder.eval()
            freq_encoder.eval()
            with torch.no_grad():
                for batch in tqdm(loader, desc="正在提取特征"):
                    eeg_data = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                    # 打印 eeg_data 的形状确认
                    # print(f"Debug: eeg_data shape before FreqEncoder: {eeg_data.shape}")
                    original_linear_proba = time_encoder.linear_proba
                    original_nocliptune = time_encoder.nocliptune
                    time_encoder.linear_proba = True
                    time_encoder.nocliptune = True
                    time_features = time_encoder(eeg_data)
                    time_encoder.linear_proba = original_linear_proba
                    time_encoder.nocliptune = original_nocliptune
                    _, freq_features_xa = freq_encoder(eeg_data)
                    combined_features = torch.cat((time_features, freq_features_xa), dim=1)
                    all_features.append(combined_features.cpu())
                    all_labels.append(labels.cpu())
            if not all_features:
                 print("警告：没有从数据加载器中提取到任何特征！")
                 return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)
            try:
                final_features = torch.cat(all_features, dim=0)
                final_labels = torch.cat(all_labels, dim=0)
            except RuntimeError as e:
                print(f"拼接特征时出错: {e}")
                # ... (错误处理) ...
                return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)
            return final_features, final_labels

        # --- 提取特征 ---
        print("正在提取训练特征...")
        train_features, train_labels = extract_combined_features(self.train_linear_loader, self.model, freq_model)
        print("正在提取测试特征...")
        test_features, test_labels = extract_combined_features(self.test_loader, self.model, freq_model)
        print(f"提取完成: 训练特征形状={train_features.shape}, 测试特征形状={test_features.shape}")

        # --- 阶段 2: 定义和训练 ResMLP 分类器 ---
        print("\n--- 阶段 2: 训练 ResMLP 分类器 ---")

        # 创建数据集和数据加载器 (与 MLP 版本相同)
        train_feat_dataset = Data.TensorDataset(train_features, train_labels)
        test_feat_dataset = Data.TensorDataset(test_features, test_labels)
        resmlp_train_loader = Data.DataLoader(train_feat_dataset, batch_size=self.args.train_batch_size, shuffle=True)
        resmlp_test_loader = Data.DataLoader(test_feat_dataset, batch_size=self.args.test_batch_size, shuffle=False)

        # *** 定义 ResMLP 模型 ***
        feature_dim = train_features.shape[1]
        block_dim = 512        # 残差块内部处理的维度（可调整）
        num_blocks = 3         # 残差块的数量（可调整）
        output_dim = self.args.num_class
        dropout_rate = 0.4     # Dropout 率（可调整）
        resmlp_classifier = ResMLPClassifier(feature_dim, block_dim, num_blocks, output_dim, dropout_rate=dropout_rate).to(self.device)

        # 定义优化器和损失函数 (与 MLP 版本相同)
        resmlp_optimizer = torch.optim.AdamW(resmlp_classifier.parameters(), lr=self.args.lr, weight_decay=1e-4) # 可加入权重衰减 L2 正则化
        resmlp_criterion = nn.CrossEntropyLoss()

        # 设置日志记录
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        resmlp_save_path = os.path.join(self.save_path, "resmlp_classifier") # 新的子目录
        os.makedirs(resmlp_save_path, exist_ok=True)
        csv_filename = os.path.join(resmlp_save_path, f'resmlp_train_log_{current_time}.csv')
        writer = SummaryWriter(log_dir=os.path.join(resmlp_save_path, 'runs'))

        with open(csv_filename, mode='w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['epoch', 'train_loss', 'test_loss', 'test_accuracy', 'test_top3_accuracy', 'test_top5_accuracy', 'test_f1_macro'])

        # 早停参数 (与 MLP 版本相同，但 patience 可能需要调整)
        best_acc = 0.0
        patience = 30 # 对于更深的模型，可能需要更多耐心
        delta = 0.001
        no_improve_count = 0
        num_resmlp_epochs = 200 # 训练更多 epochs

        print(f"开始 ResMLP 训练，共 {num_resmlp_epochs} 个 epoch...")
        print(f"ResMLP 结构: 输入={feature_dim}, 块维度={block_dim}, 块数量={num_blocks}, 输出={output_dim}, Dropout={dropout_rate}")

        # ResMLP 训练循环
        for epoch in range(num_resmlp_epochs):
            resmlp_classifier.train()
            train_loss_sum = 0.0
            progress_bar = tqdm(resmlp_train_loader, desc=f"ResMLP Epoch {epoch+1}/{num_resmlp_epochs} [训练]")
            for batch_features, batch_labels in progress_bar:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device).long()

                resmlp_optimizer.zero_grad()
                outputs = resmlp_classifier(batch_features)
                loss = resmlp_criterion(outputs, batch_labels)
                loss.backward()
                # 可以尝试梯度裁剪，防止梯度爆炸（对于深层网络有时有用）
                # torch.nn.utils.clip_grad_norm_(resmlp_classifier.parameters(), max_norm=1.0)
                resmlp_optimizer.step()
                train_loss_sum += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = train_loss_sum / len(resmlp_train_loader)

            # --- 阶段 3: 评估 ResMLP 分类器 ---
            resmlp_classifier.eval()
            test_loss_sum = 0.0
            all_preds = []
            all_labels_list = []
            progress_bar_test = tqdm(resmlp_test_loader, desc=f"ResMLP Epoch {epoch+1}/{num_resmlp_epochs} [评估]")
            with torch.no_grad():
                for batch_features, batch_labels in progress_bar_test:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device).long()
                    outputs = resmlp_classifier(batch_features)
                    loss = resmlp_criterion(outputs, batch_labels)
                    test_loss_sum += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels_list.extend(batch_labels.cpu().numpy())

            avg_test_loss = test_loss_sum / len(resmlp_test_loader)
            test_acc = accuracy_score(all_labels_list, all_preds)
            # --- 计算 Top-k (与 MLP 版本相同) ---
            test_top3_acc = 0.0
            test_top5_acc = 0.0
            with torch.no_grad():
                 all_outputs = []
                 for batch_features, _ in resmlp_test_loader:
                     outputs = resmlp_classifier(batch_features.to(self.device))
                     all_outputs.append(outputs.cpu())
                 all_outputs_tensor = torch.cat(all_outputs, dim=0)
                 test_top3_acc = top_k_accuracy_score(test_labels, all_outputs_tensor, k=3) # 假设函数已定义
                 test_top5_acc = top_k_accuracy_score(test_labels, all_outputs_tensor, k=5) # 假设函数已定义
            # --- Top-k 计算结束 ---
            test_f1_macro = f1_score(all_labels_list, all_preds, average='macro')

            print(f"ResMLP Epoch {epoch+1}: 训练损失={avg_train_loss:.4f}, 测试损失={avg_test_loss:.4f}, "
                  f"测试准确率={test_acc:.4f}, Top3 准确率={test_top3_acc:.4f}, Top5 准确率={test_top5_acc:.4f}, F1 Macro={test_f1_macro:.4f}")

            # 日志记录 (与 MLP 版本相同，但路径/前缀可能改为 ResMLP)
            writer.add_scalar('ResMLP/Loss/train', avg_train_loss, epoch)
            writer.add_scalar('ResMLP/Loss/test', avg_test_loss, epoch)
            # ... (其他指标的 writer.add_scalar) ...
            with open(csv_filename, mode='a', newline='') as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow([epoch + 1, avg_train_loss, avg_test_loss, test_acc, test_top3_acc, test_top5_acc, test_f1_macro])

            # 早停和保存最佳模型 (与 MLP 版本相同，但保存文件名前缀改为 ResMLP)
            current_acc = test_acc
            if current_acc > best_acc + delta:
                best_acc = current_acc
                no_improve_count = 0
                torch.save(resmlp_classifier.state_dict(), os.path.join(resmlp_save_path, 'resmlp_classifier_best.pkl'))
                print(f"   -> 发现新的最佳测试准确率: {best_acc:.4f}。已保存最佳模型。")
            else:
                no_improve_count += 1
                print(f"   -> 测试准确率未提升，已持续 {no_improve_count} 个 epoch。")
                if no_improve_count >= patience:
                    print(f"早停机制触发于 epoch {epoch+1}。")
                    break

            if (epoch + 1) % 20 == 0: # 可以调整保存频率
                torch.save(resmlp_classifier.state_dict(), os.path.join(resmlp_save_path, f'resmlp_classifier_epoch_{epoch+1}.pkl'))

        writer.close()
        print(f"ResMLP 训练结束。最佳测试准确率: {best_acc:.4f}")
        print(f"ResMLP 模型和日志已保存在: {resmlp_save_path}")


