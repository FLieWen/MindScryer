# # 将此代码添加到 process.py 顶部附近或单独的文件中
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
#         """
#         简单的 MLP 分类器。
#         参数:
#             input_dim (int): 输入特征的维度。
#             hidden_dims (list of int): 包含每个隐藏层大小的列表。
#             output_dim (int): 输出类别的数量。
#             dropout_rate (float): Dropout 概率。
#         """
#         super(MLPClassifier, self).__init__()
#         self.layers = nn.ModuleList() # 用于存储所有层的列表

#         last_dim = input_dim
#         # 循环构建隐藏层
#         for hidden_dim in hidden_dims:
#             self.layers.append(nn.Linear(last_dim, hidden_dim)) # 添加线性层
#             self.layers.append(nn.ReLU())                     # 添加 ReLU 激活函数
#             self.layers.append(nn.Dropout(dropout_rate))      # 添加 Dropout 层
#             last_dim = hidden_dim # 更新上一层的维度

#         # 输出层
#         self.layers.append(nn.Linear(last_dim, output_dim))
#         # 注意：这里没有最终的激活函数，CrossEntropyLoss 会自动处理 Softmax

#     def forward(self, x):
#         # 前向传播：依次通过所有层
#         for layer in self.layers:
#             x = layer(x)
#         return x



# 添加到 MLPClassifier 定义的旁边或 model 文件中
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """简单的 MLP 残差块"""
    def __init__(self, dim, hidden_dim_ratio=4, dropout_rate=0.5):
        super().__init__()
        hidden_dim = dim * hidden_dim_ratio # 隐藏层维度通常是输入/输出维度的倍数
        # 定义块内的层 F(x)
        self.layer1 = nn.Linear(dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, dim) # 输出维度必须与输入维度相同才能相加
        self.dropout2 = nn.Dropout(dropout_rate)
        # 注意：不在块内应用最后的激活函数，通常在相加后应用

    def forward(self, x):
        identity = x # 保存输入 x (捷径)
        
        # 通过块内的层 F(x)
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.dropout2(out)
        
        # 残差连接：F(x) + x
        out += identity
        # 通常在相加后应用激活函数（这里我们可以在 ResMLP 的主 forward 中应用）
        return out

class ResMLPClassifier(nn.Module):
    """带残差块的 MLP 分类器"""
    def __init__(self, input_dim, block_dim, num_blocks, output_dim, dropout_rate=0.5):
        """
        参数:
            input_dim (int): 原始输入特征维度。
            block_dim (int): 残差块内部处理的维度。
            num_blocks (int): 残差块的数量。
            output_dim (int): 输出类别的数量。
            dropout_rate (float): Dropout 概率。
        """
        super().__init__()
        
        # 输入层：将原始维度映射到块维度
        self.input_proj = nn.Linear(input_dim, block_dim)
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # 堆叠残差块
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(block_dim, dropout_rate=dropout_rate) for _ in range(num_blocks)]
        )
        # 在每个残差块后可以加激活和 Dropout
        self.block_relus = nn.ModuleList([nn.ReLU() for _ in range(num_blocks)])
        self.block_dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_blocks)])

        # 输出层：将块维度映射到类别数
        self.output_layer = nn.Linear(block_dim, output_dim)

    def forward(self, x):
        # 输入映射
        x = self.input_proj(x)
        x = self.input_relu(x)
        x = self.input_dropout(x)
        
        # 通过残差块
        for i in range(len(self.residual_blocks)):
            x = self.residual_blocks[i](x)
            x = self.block_relus[i](x) # 在块之后应用 ReLU
            x = self.block_dropouts[i](x) # 在块之后应用 Dropout
            
        # 输出层
        x = self.output_layer(x)
        # CrossEntropyLoss 会处理 Softmax
        return x
    

