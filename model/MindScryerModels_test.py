import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_
from .layers import TransformerBlock, PositionalEmbedding, CrossAttnTRMBlock
from torch.autograd import Variable
import torch.optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.utils.backcompat.broadcast_warning.enabled = True

class TimeFreqEncoder(nn.Module):
    def __init__(self, pretrained_model_time,pretrained_model_freq,args):
        super(TimeFreqEncoder, self).__init__()

        self.pretrained_model_time = pretrained_model_time
        self.pretrained_model_time.nocliptune=True
        self.pretrained_model_time.linear_proba=False
        self.pretrained_model_freq=pretrained_model_freq

        self.fc01 =nn.Linear( args.d_model+128, args.num_class)

    def forward(self,x):
        lastrep,time_feature,cls=self.pretrained_model_time(x)
        lstmcls,freq_feature=self.pretrained_model_freq(x)
        x = torch.cat((time_feature, freq_feature), dim=1)

        lastrep = x
        encoded=x
        x = self.fc01(encoded)

        scores=x
        return lastrep,encoded,scores

class TimeFreqEncoder_alpha(nn.Module):
    def __init__(self, pretrained_model_time, pretrained_model_freq, args, 
                 fusion_dim=512, learnable_alpha=True, fusion_dropout=0.5, fixed_alpha_value=0.5):
        """
        融合时间特征和频率特征的编码器。

        Args:
            pretrained_model_time: 预训练的时间编码器实例 (例如 TimeEncoder)。
                                   期望其 forward 方法返回 (..., pooled_time_feature, ...)。
            pretrained_model_freq: 预训练的频率编码器实例 (例如 EEGNetEncoder 或 FreqEncoder)。
                                   期望其 forward 方法返回 (..., mapped_freq_feature)。
            args: 包含配置的对象，至少需要:
                  args.d_model (int): TimeEncoder 的输出特征维度。
                  args.num_class (int): 最终分类的类别数。
                  args.device (str): 'cuda' 或 'cpu'。
                  (可选，用于获取默认值) args.fusion_dim, args.learnable_alpha, 
                                       args.fixed_alpha, args.fusion_dropout。
            fusion_dim (int): 时间和频率特征映射到的共同融合维度。
            learnable_alpha (bool): 融合权重 alpha 是否可学习。
            fusion_dropout (float): 在融合特征送入分类器前的 Dropout 比率。
            fixed_alpha_value (float): 如果 learnable_alpha 为 False，使用的固定 alpha 值。
        """
        super(TimeFreqEncoder_alpha, self).__init__()

        self.args = args
        self.device = args.device # 获取设备信息

        self.pretrained_model_time = pretrained_model_time
        # 确保时间模型返回需要的特征 (通常是池化后的单一向量)
        if hasattr(self.pretrained_model_time, 'nocliptune'):
            self.pretrained_model_time.nocliptune = True
        if hasattr(self.pretrained_model_time, 'linear_proba'):
            self.pretrained_model_time.linear_proba = False # 确保 forward 返回 (lastrep, pooled_feature, cls_output)
        
        self.pretrained_model_freq = pretrained_model_freq

        # --- 获取原始特征维度 ---
        self.time_feature_dim = args.d_model # 来自 TimeEncoder
        
        # 动态获取频率特征的原始维度
        if hasattr(self.pretrained_model_freq, 'output_feature_dim'): # 优先用于 EEGNetEncoder
            self.freq_feature_dim = self.pretrained_model_freq.output_feature_dim
        elif hasattr(self.pretrained_model_freq, 'output_size'): # 用于 FreqEncoder
             self.freq_feature_dim = self.pretrained_model_freq.output_size
        else:
             # 如果无法确定，可以从 args 获取或设一个默认值
             # 确保 args 中有 freq_output_dim 这个属性如果需要从这里获取
             self.freq_feature_dim = getattr(args, 'freq_output_dim', 128) 
             print(f"警告 (TimeFreqEncoder): 无法自动确定频率特征维度，使用值: {self.freq_feature_dim}")
        
        print(f"TimeFreqEncoder: Time feature dim = {self.time_feature_dim}, Freq feature dim = {self.freq_feature_dim}")

        self.fusion_dim = fusion_dim 
        print(f"TimeFreqEncoder: Target fusion dimension = {self.fusion_dim}")

        # --- 线性映射层，用于维度匹配到 fusion_dim ---
        self.map_time_to_fusion = nn.Linear(self.time_feature_dim, self.fusion_dim)
        self.map_freq_to_fusion = nn.Linear(self.freq_feature_dim, self.fusion_dim)
        
        # --- 融合权重 alpha ---
        self.learnable_alpha = learnable_alpha
        if self.learnable_alpha:
            # 定义一个可学习的标量参数 logit_alpha，初始化为0，使得初始 alpha 接近 0.5
            self.logit_alpha = nn.Parameter(torch.zeros(1)) 
            print("TimeFreqEncoder: Fusion_alpha is learnable.")
        else:
            self.alpha_value = fixed_alpha_value
            print(f"TimeFreqEncoder: Fusion_alpha is fixed to {self.alpha_value}.")
            
        # --- 融合后的处理层 (可选) ---
        self.fusion_activation = nn.ReLU() # 或者 nn.Tanh(), nn.GELU()
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        
        # --- 最终分类器 ---
        # 分类器现在作用于融合后的特征 (维度是 fusion_dim)
        self.classifier = nn.Linear(self.fusion_dim, args.num_class)
        print(f"TimeFreqEncoder: Classifier input_dim={self.fusion_dim}, output_dim={args.num_class}")


    def forward(self, x_eeg):
        """
        Args:
            x_eeg (torch.Tensor): 输入的 EEG 信号，形状应符合 pretrained_model_time 和 
                                 pretrained_model_freq 的期望。
                                 通常是 (batch_size, time_points, channels) 或 
                                 (batch_size, channels, time_points) 取决于子模型。
        Returns:
            tuple: (last_representation, encoded_representation, scores)
                   last_representation: 通常是融合后的特征或特定模态的特征。
                   encoded_representation: 融合后的特征。
                   scores: 最终的分类 logits。
        """
        # 1. 提取时间和频率特征
        # 确保子模型在其期望的设备上 (如果它们没有在 __init__ 中自动处理)
        # self.pretrained_model_time.to(x_eeg.device)
        # self.pretrained_model_freq.to(x_eeg.device)

        # 调用 TimeEncoder，期望返回 (全序列输出, 池化/聚合特征, 分类头输出)
        # 我们需要第二个返回值作为 time_feature
        try:
            _, time_feature_pooled, _ = self.pretrained_model_time(x_eeg)
        except Exception as e:
            print(f"错误: 调用 pretrained_model_time 时出错: {e}")
            print(f"输入 x_eeg 形状: {x_eeg.shape}")
            raise e

        # 调用 FreqEncoder/EEGNetEncoder，期望返回 (分类头输出, 特征表示)
        # 我们需要第二个返回值作为 freq_feature
        try:
            # EEGNetEncoder: forward 返回 (logits, mapped_features)
            # FreqEncoder: forward 返回 (logits, xa_features)
            _, freq_feature_mapped = self.pretrained_model_freq(x_eeg)
        except Exception as e:
            print(f"错误: 调用 pretrained_model_freq 时出错: {e}")
            print(f"输入 x_eeg 形状: {x_eeg.shape}")
            raise e

        # 2. 将特征映射到相同的融合维度
        mapped_time_feature = self.map_time_to_fusion(time_feature_pooled)
        mapped_freq_feature = self.map_freq_to_fusion(freq_feature_mapped)

        # 3. (可选) 在映射后应用激活函数
        mapped_time_feature = self.fusion_activation(mapped_time_feature)
        mapped_freq_feature = self.fusion_activation(mapped_freq_feature)

        # 4. 计算融合权重 alpha
        if self.learnable_alpha:
            current_alpha = torch.sigmoid(self.logit_alpha.to(mapped_time_feature.device)) # 确保 alpha 在正确设备
            # 打印 alpha 值 (用于调试，可以限制打印频率)
            # if self.training and torch.rand(1).item() < 0.001: # 训练时小概率打印
            #     print(f"Learned alpha: {current_alpha.item():.4f}")
        else:
            current_alpha = self.alpha_value

        # 5. 加权融合 (确保 alpha 可以广播)
        # current_alpha 可能是一个标量张量 torch.tensor([alpha_val])
        # 或者是一个 Python float，需要转换为 tensor
        if not isinstance(current_alpha, torch.Tensor):
            current_alpha = torch.tensor(current_alpha, device=mapped_time_feature.device, dtype=mapped_time_feature.dtype)
        
        # 如果 current_alpha 是一个标量张量，它会自动广播
        fused_feature = current_alpha * mapped_time_feature + (1 - current_alpha) * mapped_freq_feature

        # 6. (可选) 在融合后应用 Dropout
        fused_feature_processed = self.fusion_dropout(fused_feature)
        
        # 7. 通过最终分类器得到分数
        scores = self.classifier(fused_feature_processed)

        # 定义 lastrep 和 encoded，通常可以是融合后的特征
        last_representation = fused_feature_processed # 或者使用融合前的 fused_feature
        encoded_representation = fused_feature_processed

        return last_representation, encoded_representation, scores



def _get_activation(name):
    act_map = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'leakyrelu': nn.LeakyReLU,
    }
    return act_map.get(name, nn.Tanh)


class AlignNet(nn.Module):
    """Configurable AlignNet for EEG-to-CLIP alignment.

    Parameters
    ----------
    input_size : int        TimeEncoder d_model (e.g. 1024)
    freq_size : int         FreqEncoder output dim (e.g. 128)
    output_size : int       CLIP space dim (e.g. 77*768 = 59136)
    pretrained_model : nn.Module   TimeFreqEncoder instance
    num_blocks : int        Number of residual blocks (0-4, default 3)
    expansion : int         Hidden expansion factor (default 4)
    activation : str        'tanh' / 'relu' / 'gelu' / 'leakyrelu'
    use_classifier_logits : bool   Whether to concat 40-dim classifier scores into input
    """

    def __init__(self, input_size, freq_size, output_size, pretrained_model,
                 num_blocks=3, expansion=4, activation='tanh',
                 use_classifier_logits=True):
        super(AlignNet, self).__init__()

        self.pretrained_model = pretrained_model
        self.num_blocks = num_blocks
        self.use_classifier_logits = use_classifier_logits
        hidden_dim = input_size
        act_fn = _get_activation(activation)

        in_dim = (input_size + freq_size + 40) if use_classifier_logits else (input_size + freq_size)
        exp_dim = expansion * hidden_dim

        # --- build expand / compress layers ---
        self.expand_layers = nn.ModuleList()
        self.compress_layers = nn.ModuleList()
        self.expand_acts = nn.ModuleList()
        self.compress_acts = nn.ModuleList()

        for i in range(num_blocks + 1):          # num_blocks+1 expands
            in_ch = in_dim if i == 0 else hidden_dim
            self.expand_layers.append(nn.Linear(in_ch, exp_dim))
            self.expand_acts.append(act_fn())

            if i < num_blocks:                   # num_blocks compresses
                self.compress_layers.append(nn.Linear(exp_dim, hidden_dim))
                self.compress_acts.append(act_fn())

        self.fc_out = nn.Linear(exp_dim, output_size)

    def forward(self, x):
        lastrep, encoded, scores = self.pretrained_model(x)

        if self.use_classifier_logits:
            x = torch.cat((encoded, scores), dim=1)
        else:
            x = encoded

        prev_expand = None
        prev_compress = None

        for i in range(len(self.expand_layers)):
            x = self.expand_layers[i](x)
            x = self.expand_acts[i](x)

            if prev_expand is not None and i >= 2:
                x = x + prev_expand
            prev_expand = x

            if i < len(self.compress_layers):
                x = self.compress_layers[i](x)
                x = self.compress_acts[i](x)

                if prev_compress is not None and i >= 1:
                    x = x + prev_compress
                prev_compress = x

        return self.fc_out(x)

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter

        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x

class Tokenizer(nn.Module):
    def __init__(self, rep_dim, vocab_size):
        super(Tokenizer, self).__init__()
        self.center = nn.Linear(rep_dim, vocab_size)

    def forward(self, x):
        bs, length, dim = x.shape
        probs = self.center(x.view(-1, dim))
        ret = F.gumbel_softmax(probs)
        indexes = ret.max(-1, keepdim=True)[1]

        return indexes.view(bs, length)

class Regressor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)

        return rep_mask_token

class ChannelMapping(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ChannelMapping, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, int(input_dim+output_dim)//2, 1)
        self.conv2 = nn.Conv1d(int(input_dim+output_dim)//2, output_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class TimeEncoder(nn.Module):
    def __init__(self, args):
        super(TimeEncoder, self).__init__()
        d_model = args.d_model  
        self.d=d_model  # Transformer特征维度
        self.momentum = args.momentum   # 
        self.linear_proba = True
        self.nocliptune=True
        self.device = args.device
        self.data_shape = args.data_shape
        self.max_len = int(self.data_shape[0] / args.wave_length)
        print(self.max_len)
        self.mask_len = int(args.mask_ratio * self.max_len)
        self.position = PositionalEmbedding(self.max_len, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model, ))
        self.input_projection = nn.Conv1d(args.data_shape[1], d_model, kernel_size=args.wave_length,
                                          stride=args.wave_length)
        self.encoder = TransformerEncoder(args)
        self.momentum_encoder = TransformerEncoder(args)
        self.tokenizer = Tokenizer(d_model, args.vocab_size)
        self.reg = Regressor(d_model, args.attn_heads, 4 * d_model, 1, args.reg_layers)
        self.predict_head = nn.Linear(d_model, args.num_class)
        self.channelmapping=ChannelMapping(self.max_len,77)
        self.dimmapping = nn.Linear(d_model, 768)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def pretrain_forward(self, x):
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        tokens = self.tokenizer(x)

        x += self.position(x)

        rep_mask_token = self.mask_token.repeat(x.shape[0], x.shape[1], 1) + self.position(x)

        index = np.arange(x.shape[1])
        random.shuffle(index)
        v_index = index[:-self.mask_len]
        m_index = index[-self.mask_len:]
        visible = x[:, v_index, :]
        mask = x[:, m_index, :]
        tokens = tokens[:, m_index]

        rep_mask_token = rep_mask_token[:, m_index, :]

        rep_visible = self.encoder(visible)
        with torch.no_grad():
            rep_mask = self.momentum_encoder(mask)

        rep_mask_prediction = self.reg(rep_visible, rep_mask_token)
        token_prediction_prob = self.tokenizer.center(rep_mask_prediction)

        return [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens]

    def forward(self, x):
        if self.linear_proba==True and self.nocliptune==True:
            #with torch.no_grad():
            x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
            x += self.position(x)
            x = self.encoder(x)
            return torch.mean(x, dim=1)

        if self.linear_proba==False and self.nocliptune==True:
            x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
            x += self.position(x)
            x = self.encoder(x)
            #lastrep=torch.mean(x, dim=1)
            lastrep=x
            xcls=self.predict_head(torch.mean(x, dim=1))
            return lastrep, torch.mean(x, dim=1), xcls

        if self.nocliptune == False: #CLIP
            x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
            x += self.position(x)
            x = self.encoder(x)
            lastrep=torch.mean(x, dim=1)
            x=self.channelmapping(x)
            x = self.dimmapping(x)

            return lastrep#,x

    def get_tokens(self, x):
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        tokens = self.tokenizer(x)
        return tokens

class FreqEncoder(nn.Module):

    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size

        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)
        self.classifier = nn.Linear(output_size, 40)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        x = x.cpu()
        fourier_transform = np.fft.fft(x, axis=2)
        half_spectrum = fourier_transform[:, :, 1:440 // 2 + 1]
        amplitude_spectrum = np.abs(half_spectrum)

        amplitude_spectrum = torch.tensor(amplitude_spectrum).float()

        x = amplitude_spectrum.permute(0, 2, 1)
        x = x.to("cuda")

        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size),
                     torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))

        x = self.lstm(x, lstm_init)[0][:, -1, :]
        reps = x
        # Forward output
        xa = F.relu(self.output(x))
        x = self.classifier(xa)
        return x, xa


class EEGNet(nn.Module):
    """
    简化的 EEGNet 实现 (基于原始论文)
    输入形状: (batch_size, 1, num_channels, num_time_points)
    """
    def __init__(self, num_channels, num_time_points, num_classes, F1=8, D=2, F2=16, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        self.num_channels = num_channels
        self.num_time_points = num_time_points
        self.num_classes = num_classes

        # Block 1: Temporal Convolution + Depthwise Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False) # 时间卷积
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False) # 深度卷积 (跨通道)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Convolution (Depthwise + Pointwise)
        self.separable_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), groups=F1*D, bias=False) # 深度卷积部分
        self.pointwise_conv = nn.Conv2d(F2, F2, (1, 1), bias=False) # 逐点卷积部分 (合并上面两步等效于 SeparableConv)
        # 注意：更标准的 SeparableConv 实现方式如下:
        # self.separable_conv = nn.Sequential(
        #     nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False), # Depthwise
        #     nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False) # Pointwise
        # )
        self.bn3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Flatten and Classifier
        self.flatten = nn.Flatten()
        # 计算展平后的特征维度
        self.final_feature_dim = self._get_final_feature_dim(num_channels, num_time_points, F1, D, F2)
        self.fc = nn.Linear(self.final_feature_dim, num_classes)

    def _get_final_feature_dim(self, Chans, Samples, F1, D, F2):
        # 模拟前向传播来计算维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, Chans, Samples)
            x = self.pool1(self.activation1(self.bn2(self.depthwise_conv(self.bn1(self.conv1(dummy_input))))))
            # 模拟 SeparableConv (使用简化的两步)
            x_sep = self.separable_conv(x)
            x_point = self.pointwise_conv(x_sep)
            x = self.pool2(self.activation2(self.bn3(x_point)))
            x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        # x 初始形状: (batch_size, num_channels, num_time_points)
        # EEGNet 需要 (batch_size, 1, num_channels, num_time_points)
        if x.dim() == 3:
            x = x.unsqueeze(1) # 添加虚拟的 '1' 维度

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2 (使用简化的两步 separable)
        x_sep = self.separable_conv(x)
        x = self.pointwise_conv(x_sep)
        # 如果使用 nn.Sequential 实现 SeparableConv:
        # x = self.separable_conv(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten and Classify
        features = self.flatten(x)
        logits = self.fc(features)

        return logits, features # 返回 logits 和 flatten 之前的特征
    

# 创建一个包装类，使其接口与原 FreqEncoder 相似
class EEGNetEncoder(nn.Module):
    def __init__(self, input_channels, input_time_points, num_classes=40, # 与原 FreqEncoder 的分类器匹配
                 F1=8, D=2, F2=16, dropout_rate=0.5, output_feature_dim=128): # 添加 output_feature_dim
        super().__init__()
        self.eegnet = EEGNet(num_channels=input_channels,
                             num_time_points=input_time_points,
                             num_classes=num_classes,
                             F1=F1, D=D, F2=F2, dropout_rate=dropout_rate)

        # 添加一个可选的线性层，将 EEGNet 提取的特征映射到所需的维度 (类似原 FreqEncoder 的 self.output)
        self.feature_mapping = nn.Linear(self.eegnet.final_feature_dim, output_feature_dim)
        self.activation = nn.ReLU() # 保持与原 FreqEncoder 一致

        # 分类器现在使用 EEGNet 内部的 fc 层
        # 如果需要不同的分类器，可以在这里重新定义

    def forward(self, x):
        # 假设调用者传入的 x 形状是 (batch_size, time_points, channels)
        # 即 (B, T, C) -> (128, 440, 128)

        # --- 添加维度检查和调整 ---
        # EEGNet 期望输入 (B, C, T)
        expected_channels = self.eegnet.num_channels
        expected_time_points = self.eegnet.num_time_points

        # 检查输入形状是否符合预期 (B, T, C)
        if x.shape[1] == expected_time_points and x.shape[2] == expected_channels:
            # 如果是 (B, T, C)，则 permute 为 (B, C, T)
            x = x.permute(0, 2, 1)
            # print(f"Debug: EEGNetEncoder permuted input to: {x.shape}") # 形状应为 (128, 128, 440)
        elif x.shape[1] == expected_channels and x.shape[2] == expected_time_points:
            # 如果已经是 (B, C, T)，则无需操作
            # print(f"Debug: EEGNetEncoder input already has shape (B, C, T): {x.shape}")
            pass
        else:
            # 形状不匹配，抛出错误
            raise ValueError(f"EEGNetEncoder 输入形状不匹配! 期望 (B, {expected_time_points}, {expected_channels}) 或 (B, {expected_channels}, {expected_time_points})，但得到 {x.shape}")
        # --- 维度调整结束 ---

        # 现在 x 的形状应该是 (B, C, T)，即 (128, 128, 440)
        logits, features = self.eegnet(x) # 获取 EEGNet 的输出

        # print(f"Debug: features shape from EEGNet: {features.shape}") # 应该显示 (B, 约208)

        # 将 EEGNet 的 'features' (flatten 后的) 映射到所需的输出维度
        # 此时 features 形状应为 (B, 208)
        # print(f"Debug: features shape after eegnet: {features.shape}")
        mapped_features = self.activation(self.feature_mapping(features)) # 对应原 FreqEncoder 的 xa
     
        # print(f"Debug: mapped_features shape from EEGNetEncoder: {mapped_features.shape}") # 应该显示 (B, 128)
 
        # 返回 EEGNet 的分类 logits 和映射后的特征 (对应 xa)
        return logits, mapped_features
