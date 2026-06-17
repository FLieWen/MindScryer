import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch.optim as optim # 导入 optim
from torch.utils.data import TensorDataset, DataLoader as TensorDataLoader
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold # 导入 SVM 相关
from sklearn.svm import SVC # 导入 SVM 分类器

def get_rep_with_label(model, dataloader):      # 提取EEG信号特征，并返回特征与标签
    reps = []   # 存储提取特征
    labels = []     # 存储样本标签
    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq, label,clip,clip_moreinf,neg_image,neg_text = batch
            seq = seq.to(args.device)
            labels += label.cpu().numpy().tolist()
            rep = model(seq)    # 通过模型提取特征表示
            reps += rep.cpu().numpy().tolist()
    return reps, labels     # 返回提取的特征和对应的标签

def get_freqrep_with_label(freqtime_model, dataloader):
    reps = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq, label,clip_moreinf = batch
            seq = seq.to(args.device)
            labels += label.cpu().numpy().tolist()
            rep,encoded,xcls = freqtime_model(seq)
            reps += rep.cpu().numpy().tolist()
    return reps, labels

def fit_lr(features, y):
    pipe = make_pipeline(
        StandardScaler(),   # 对特征进行标准化
        LogisticRegression(     # 逻辑回归分类器，用于多类别分类
            random_state=3407,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)   # 使用给定的特征features和标签y训练模型
    return pipe

def get_rep_with_label_with_image_name(model, dataloader):  # 提取EEG信号的特征
    reps = []
    clips=[]
    last_reps=[]
    labels = []
    preds= []
    seqs=[]
    scores= []
    image_names=[]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq, label,image_name = batch
            seqs+=seq.cpu().numpy().tolist()
            seq = seq.to(args.device)
            labels += label.cpu().numpy().tolist()
            rep, encoded,score = model(seq)     # 通过模型提取特征rep
            reps+=rep.cpu().numpy().tolist()
            image_names+=list(image_name)
            _, pred = torch.topk(score, 1)
            preds += pred.cpu().numpy().tolist()
        acc=accuracy_score(y_true=labels, y_pred=preds)     # 计算准确度
        print("testortrainacc")
        print(acc)

    return labels,image_names, preds,seqs,reps,acc


class MLPClassifierTorch(nn.Module):
    """一个简单的基于 PyTorch 的 MLP 分类器"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        """
        Args:
            input_dim (int): 输入特征的维度 (例如 time_size + freq_size)
            hidden_dims (list of int): 隐藏层的维度列表，例如 [512, 256]
            output_dim (int): 输出维度 (类别数量)
            dropout_rate (float): Dropout 比率
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        current_dim = input_dim
        # 添加隐藏层
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) # 可选：添加 BatchNorm 稳定训练
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        
        # 添加输出层
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征，形状 (batch_size, input_dim)
        Returns:
            torch.Tensor: 输出 logits，形状 (batch_size, output_dim)
        """
        return self.classifier(x)
    

# --- 定义 fit_mlp 函数 (可以放在 classification.py 中然后导入) ---
def fit_mlp(features_train, y_train, features_val, y_val, input_dim, num_classes,
            hidden_dims=[1024, 512, 256], dropout=0.5, lr=1e-4, epochs=50, batch_size=256,
            patience=30, device='cuda', run_name_prefix=""):
    """训练 MLP 分类器"""
    print(f"\n--- 开始训练 {run_name_prefix}MLP 分类器 ---")
    print(f"MLP 结构: Input={input_dim}, Hidden={hidden_dims}, Output={num_classes}, Dropout={dropout}")
    print(f"训练参数: LR={lr}, Epochs={epochs}, BatchSize={batch_size}, Patience={patience}")

    # 1. 数据标准化
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_val_scaled = scaler.transform(features_val)

    # 2. 转换为 Tensor 和 DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(features_train_scaled), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(features_val_scaled), torch.LongTensor(y_val))
    # 使用 TensorDataLoader 避免与主 DataLoader 冲突
    train_loader = TensorDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TensorDataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    # 3. 初始化模型、损失、优化器
    model_mlp = MLPClassifierTorch(input_dim, hidden_dims, num_classes, dropout_rate=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=patience // 2, verbose=True)

    # 4. 训练循环与早停
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model_mlp.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        progress_bar_train = tqdm(train_loader, desc=f"{run_name_prefix}MLP Epoch {epoch+1}/{epochs} [训练]", leave=False)
        for inputs, labels in progress_bar_train:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            progress_bar_train.set_postfix(loss=loss.item())
        avg_train_loss = train_loss_sum / train_total if train_total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0

        # 验证
        model_mlp.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_mlp(inputs)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss_sum / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"{run_name_prefix}MLP Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        scheduler.step(val_acc)

        # 早停与保存最佳
        if val_acc > best_val_acc:
            print(f"  -> {run_name_prefix}验证集准确率提升: {best_val_acc:.4f} -> {val_acc:.4f}. 保存模型...")
            best_val_acc = val_acc
            best_model_state = model_mlp.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  -> {run_name_prefix}早停触发：验证集准确率连续 {patience} epochs 未提升。")
                break

    if best_model_state:
        model_mlp.load_state_dict(best_model_state)
    print(f"--- {run_name_prefix}MLP 训练结束。最佳验证准确率: {best_val_acc:.4f} ---")
    return model_mlp, scaler # 返回模型和 scaler

def fit_mlp_no_val(features_train, y_train, input_dim, num_classes,
                   hidden_dims=[512, 256], dropout=0.5, lr=1e-4, epochs=50,
                   batch_size=256, device='cuda', run_name_prefix="", weight_decay=0):
    """训练 MLP 分类器，不使用验证集，但记录训练过程中损失最低的模型状态"""
    print(f"\n--- 开始训练 {run_name_prefix}MLP 分类器 (追踪最佳训练损失) ---")
    print(f"MLP 结构: Input={input_dim}, Hidden={hidden_dims}, Output={num_classes}, Dropout={dropout}")
    print(f"训练参数: LR={lr}, Epochs={epochs}, BatchSize={batch_size}, WeightDecay={weight_decay}")

    # 1. 数据标准化
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    print("训练特征已标准化。")

    # 2. 转换为 PyTorch Tensor 和 DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(features_train_scaled), torch.LongTensor(y_train))
    train_loader = TensorDataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # drop_last=True 避免BatchNorm问题

    if len(train_loader) == 0:
        print(f"警告: ({run_name_prefix}MLP) 训练样本数不足以形成一个完整批次。返回未训练模型。")
        model_mlp_dummy = MLPClassifierTorch(input_dim, hidden_dims, num_classes, dropout_rate=dropout).to(device)
        return model_mlp_dummy, scaler

    # 3. 初始化模型、损失函数、优化器
    model_mlp = MLPClassifierTorch(input_dim, hidden_dims, num_classes, dropout_rate=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model_mlp.parameters(), lr=lr, weight_decay=weight_decay) # 假设使用AdamW

    # --- 新增：追踪最佳模型状态 (基于训练损失) ---
    best_train_loss = float('inf')
    best_model_state_on_train = None
    # ---

    # 4. 训练循环
    for epoch in range(epochs):
        model_mlp.train()
        epoch_train_loss_sum = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        progress_bar_train = tqdm(train_loader, desc=f"{run_name_prefix}MLP Epoch {epoch+1}/{epochs} [训练]", leave=False)
        
        for inputs, labels in progress_bar_train:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss_sum += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            epoch_train_total += labels.size(0)
            epoch_train_correct += (predicted == labels).sum().item()
            progress_bar_train.set_postfix(loss=loss.item())
        
        avg_epoch_train_loss = epoch_train_loss_sum / epoch_train_total if epoch_train_total > 0 else float('inf')
        epoch_train_acc = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0.0
        print(f"{run_name_prefix}MLP Epoch {epoch+1}: Train Loss={avg_epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f}")

        # --- 更新最佳模型状态 (基于当前 epoch 的平均训练损失) ---
        if avg_epoch_train_loss < best_train_loss:
            best_train_loss = avg_epoch_train_loss
            best_model_state_on_train = model_mlp.state_dict() # 保存当前模型的状态字典
            print(f"  -> 新的最佳训练损失: {best_train_loss:.4f} (在 Epoch {epoch+1})。记录此模型状态。")
        # ---

    print(f"--- {run_name_prefix}MLP 训练结束。最终训练准确率: {epoch_train_acc:.4f} ---")
    
    # 加载在训练过程中记录的最佳模型状态
    if best_model_state_on_train:
        print(f"加载训练过程中最佳模型状态 (基于最低训练损失: {best_train_loss:.4f})...")
        model_mlp.load_state_dict(best_model_state_on_train)
    else:
        # 如果由于某种原因没有记录到最佳状态 (例如 epochs=0 或训练数据为空)，则返回最后一个 epoch 的模型
        print(f"警告: 未记录到最佳训练模型状态，返回最后一个 epoch 的模型。")
        
    return model_mlp, scaler

# --- 定义 fit_svm 函数 ---
def fit_svm(features_train, y_train, kernel='rbf', C=1.0, gamma='scale',
            perform_grid_search=False, cv_folds=5):
    """训练 SVM 分类器，可选进行网格搜索"""
    print(f"\n--- 开始训练 SVM 分类器 ---")
    print(f"参数: kernel={kernel}, C={C}, gamma={gamma}, GridSearch={perform_grid_search}")

    # 1. 数据标准化 (非常重要!)
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    print("训练特征已标准化。")

    # 2. 初始化 SVM 模型
    base_svm = SVC(probability=True, random_state=3407) # probability=True 以便后续获取概率

    if perform_grid_search:
        print(f"执行网格搜索 (CV={cv_folds}折)...")
        # 定义要搜索的参数网格 (可以根据需要调整范围)
        param_grid = {
            'C': [0.1, 1, 10, 50, 100], # 正则化参数
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001], # RBF/poly/sigmoid 核系数
            'kernel': ['rbf', 'linear'] # 尝试不同的核
        }
        # 使用分层 K 折交叉验证
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=3407)
        # scoring 可以是 'accuracy', 'f1_macro' 等
        grid_search = GridSearchCV(base_svm, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1) # n_jobs=-1 使用所有 CPU 核心
        
        grid_search.fit(features_train_scaled, y_train)
        
        print(f"网格搜索完成。最佳参数: {grid_search.best_params_}")
        print(f"对应的最佳交叉验证准确率: {grid_search.best_score_:.4f}")
        svm_clf = grid_search.best_estimator_ # 使用找到的最佳模型
    else:
        print("使用默认/指定的参数训练 SVM...")
        svm_clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=3407)
        svm_clf.fit(features_train_scaled, y_train)
        print("SVM 模型训练完成。")

    print(f"--- SVM 训练结束 ---")
    # 返回训练好的 SVM 模型和标准化器
    return svm_clf, scaler