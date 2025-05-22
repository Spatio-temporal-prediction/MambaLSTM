import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
import os,sys
import numpy as np
from sklearn.metrics import average_precision_score
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import average_precision_score
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score  # 新增导入
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score

#修改环境变量，添加项目目录，方便不同路径的脚本模块调用
curPath = os.path.abspath(os.path.dirname(__file__))
print(curPath)
sys.path.append(curPath)
from ICDM.ICDM_MambaLSTM import ICDM_MambaLSTM


from data_process import dataloader

import logging
import datetime
import torch
torch.autograd.set_detect_anomaly(True)
def log_init():

    log_dir = "log_ICDM"
    os.makedirs(log_dir, exist_ok=True)  # 自动创建目录（如果不存在）
    
    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)s - %(message)s'  # 仅保留日志级别和消息
    )
    # formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_{timestamp}.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台输出（可选）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

log_init()

def evaluate_single_step(pred, true, L=None):
    """
    单时间步评估函数（batch_size, num_nodes）

    参数:
        pred: 预测张量 (batch_size, num_nodes)
        true: 真实值张量 (batch_size, num_nodes)
        L: 用于 Acc@L 的高风险区域大小，None 时仅计算 RMSE 和 MAP

    返回:
        metrics: 包含 RMSE, Acc@L, MAP 的字典
    """
    device = pred.device
    batch_size, num_nodes = pred.shape

    # 转换为 numpy 进行计算
    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()

    # 初始化结果字典
    metrics = {}

    # 计算 RMSE（均方根误差）
    rmse_error = np.mean((pred_np - true_np) ** 2)
    metrics['RMSE'] = np.sqrt(rmse_error)

    # 计算Acc@L（仅关注高风险区域Top - L）
    if L is not None:
        acc_list = []
        for b in range(batch_size):
            # 获取预测和真实的top L索引
            pred_top = np.argsort(-pred_np[b])[:L]
            print(pred_top)
            true_top = np.argsort(-true_np[b])[:L]
            print(true_top)
            intersection = np.intersect1d(pred_top, true_top)
            print(intersection)
            sys.exit()
            acc_list.append(len(intersection) / L)
        metrics[f'Acc@{L}'] = np.mean(acc_list)


    # 计算 MAP（不再限制前 L 个高风险区域，而是计算整体 MAP）
    map_list = []
    for b in range(batch_size):
        y_true = true_np[b]  # 真实值
        y_score = pred_np[b]  # 预测值
        
        # 计算 AP（如果 y_true 只有一个类别，AP 计算会出错，所以做检查）
        if np.unique(y_true).size > 1:
            ap = average_precision_score(y_true, y_score)
        else:
            ap = 0.0  # 避免计算出错，AP 设为 0

        map_list.append(ap)

    metrics['MAP'] = np.mean(map_list)

    return metrics

def calculate_metrics(pred, true):
    """
    计算二分类任务的 F1 Score, Accuracy, AUC-PR, AUC-ROC, Precision, Recall
    
    参数:
        pred (torch.Tensor): 模型输出的预测概率（形状任意，会被展平）
        true (torch.Tensor): 真实标签（0或1，形状任意，会被展平）
    
    返回:
        dict: 包含各项指标的字典
    """
    # 转换张量到numpy并脱离计算图
    pred_np = pred.detach().cpu().numpy().flatten()
    true_np = true.detach().cpu().numpy().flatten()

    # 初始化指标字典
    metrics = {}

    # 处理AUC-ROC可能出现的异常（如单一类别）
    try:
        metrics["AUC-ROC"] = roc_auc_score(true_np, pred_np)
    except ValueError:
        metrics["AUC-ROC"] = float('nan')

    # 计算AUC-PR（此指标在无正样本时自动返回0）
    metrics["AUC-PR"] = average_precision_score(true_np, pred_np)

    # 二值化预测值并计算分类指标
    pred_binary = (pred_np >= 0.5).astype(np.int32)
    
    # 添加 Precision 和 Recall（处理 zero_division 警告）
    metrics["Accuracy"] = accuracy_score(true_np, pred_binary)
    metrics["Precision"] = precision_score(true_np, pred_binary, zero_division=0)  # 新增
    metrics["Recall"] = recall_score(true_np, pred_binary, zero_division=0)         # 新增
    metrics["F1 Score"] = f1_score(true_np, pred_binary, zero_division=0)           # 修复原有代码的潜在警告

    return metrics

def calculate_rmse(pred, true):
    """
    计算均方根误差（RMSE）
    """
    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()
    rmse_error = np.mean((pred_np - true_np) ** 2)
    return np.sqrt(rmse_error)

def calculate_acc_at_L(pred, true, L):
    """
    计算Acc@L指标
    """
    batch_size, num_nodes = pred.shape
    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()
    
    acc_list = []
    for b in range(batch_size):
        # 获取预测和真实的top L索引
        pred_top = np.argsort(-pred_np[b])[:L]
        true_top = np.argsort(-true_np[b])[:L]
        intersection = np.intersect1d(pred_top, true_top)
        acc_list.append(len(intersection) / L)
    return np.mean(acc_list)

def calculate_acc_at_dynamic_L(pred, true):
    """
    计算动态Acc@L指标，L为每个样本的真实正标签数量
    """
    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()
    batch_size, num_nodes = pred_np.shape
    
    acc_list = []
    for b in range(batch_size):
        # 获取当前样本的真实正标签数量
        true_labels = true_np[b]
        L = int(np.sum(true_labels))  # 真实正样本数作为L
        
        # 处理无正样本的情况（分母为0时跳过）
        if L == 0:
            acc_list.append(0.0)  # 或根据需求改为跳过 continue
            continue
        
        # 获取预测的Top-L索引（根据预测概率排序）
        pred_top = np.argsort(-pred_np[b])[:L]
        
        # 获取真实的Top-L索引（直接取所有正样本）
        true_top = np.where(true_labels == 1)[0]  # 等价于真实Top-L
        
        # 计算交集比例
        intersection = np.intersect1d(pred_top, true_top)
        acc = len(intersection) / L
        acc_list.append(acc)
    
    return np.mean(acc_list) if acc_list else 0.0  # 空列表返回0

def calculate_dynamic_map(pred, true):
    """
    动态调整L值的MAP计算函数
    
    参数：
    pred : torch.Tensor - 模型预测分数 [batch_size, num_nodes]
    true : torch.Tensor - 真实标签（0/1）[batch_size, num_nodes]
    
    返回：
    float - 有效样本的平均MAP值
    """
    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()
    batch_size, num_nodes = pred_np.shape
    
    map_list = []
    
    for b in range(batch_size):
        # 获取当前样本数据
        pred_scores = pred_np[b]
        true_labels = true_np[b]
        
        # 计算动态L值（真实事故数量）
        L = int(np.sum(true_labels))
        if L == 0:
            continue  # 跳过无事故样本
        
        # 获取预测的Top L区域
        pred_top = np.argsort(-pred_scores)[:L]  # 降序排列取前L个
        
        # 获取真实事故区域
        true_positives = np.where(true_labels == 1)[0]
        true_set = set(true_positives)
        
        # 计算平均精度
        cumulative_correct = 0
        precision_sum = 0.0
        
        for rank, node in enumerate(pred_top, 1):
            # 计算当前节点是否命中
            hit = 1 if node in true_set else 0
            
            # 更新累计正确数
            cumulative_correct += hit
            
            # 计算当前精度
            current_precision = cumulative_correct / rank
            
            # 累加加权精度
            precision_sum += current_precision * hit
        
        # 计算单个样本的AP
        ap = precision_sum / L if L > 0 else 0.0
        map_list.append(ap)
    
    # 处理全batch无事故的特殊情况
    if not map_list:
        print("Warning: All samples have zero true accidents")
        return 0.0
    
    return np.mean(map_list)

def calculate_map(pred, true, L):
    """
    计算Mean Average Precision (MAP@L)
    
    参数：
    pred : torch.Tensor - 模型预测分数 [batch_size, num_nodes]
    true : torch.Tensor - 真实分数 [batch_size, num_nodes]
    L : int - 需要评估的Top L排名长度
    
    返回：
    float - 整个batch的平均MAP值
    """
    # 转换为CPU numpy数组
    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()
    batch_size, num_nodes = pred_np.shape
    
    # 结果存储列表
    map_list = []
    
    for b in range(batch_size):
        # 获取当前样本的预测和真实值
        pred_scores = pred_np[b]
        true_scores = true_np[b]
        
        # 生成预测和真实的Top L索引
        pred_topL = np.argsort(-pred_scores)[:L]  # 降序排列取前L
        true_topL = np.argsort(-true_scores)[:L]
        true_set = set(true_topL)
        
        # 初始化累计变量
        cumulative_correct = 0
        sum_precision = 0.0
        
        for position in range(L):
            current_node = pred_topL[position]
            
            # 计算real(j)
            real = 1 if current_node in true_set else 0
            
            # 更新累计正确数
            cumulative_correct += real
            
            # 计算pre(j) = 正确数 / 当前位置(从1开始)
            denominator = position + 1  # 避免除零错误
            precision_at_j = cumulative_correct / denominator if denominator !=0 else 0.0
            
            # 累加精度贡献
            sum_precision += precision_at_j * real
        
        # 计算当前样本的AP
        ap = sum_precision / L
        map_list.append(ap)
    
    # 返回整个batch的平均MAP
    return np.mean(map_list)

# 创建数据集对象
class TimeSeriesDataset(Dataset):
    def __init__(self, grid_X_c, grid_y_c, grid_X_f, grid_y_f, node_X_c, node_X_f, target_time_feature):
        self.X_c = torch.tensor(grid_X_c, dtype=torch.float32)
        self.y_c = torch.tensor(grid_y_c, dtype=torch.float32)
        self.X_f = torch.tensor(grid_X_f, dtype=torch.float32)
        self.y_f = torch.tensor(grid_y_f, dtype=torch.float32)
        self.node_X_c = torch.tensor(node_X_c, dtype=torch.float32)
        self.node_X_f = torch.tensor(node_X_f, dtype=torch.float32)
        self.target_time_feature = torch.tensor(target_time_feature, dtype=torch.float32)
        # self.flag_c = torch.tensor(flag_c, dtype=torch.float32)
        # self.flag_f = torch.tensor(flag_f, dtype=torch.float32)

    def __len__(self):
        return len(self.X_c)

    def __getitem__(self, idx):
        return self.X_c[idx], self.y_c[idx], self.X_f[idx], self.y_f[idx], self.node_X_c[idx], self.node_X_f[idx], self.target_time_feature[idx]


# 加载数据
#目录
data_path = os.path.join(os.path.split(curPath)[0], "data",'npy_new_nodiff')
#时空数据：（T,D,H,W）
all_data_c_path = os.path.join(data_path, 'new_grid_data_c_4d.npy')
all_data_f_path = os.path.join(data_path, 'new_grid_data_f_4d.npy')

#映射矩阵：（HW,valid）
grid_node_map_c_path = os.path.join(data_path, 'new_grid_node_map_c.npy')
grid_node_map_f_path = os.path.join(data_path, 'new_grid_node_map_f.npy')
#静态特征
static_feat_c_path = os.path.join(data_path, 'new_static_feat_c.npy')
static_feat_f_path = os.path.join(data_path, 'new_static_feat_f.npy')

#语义边
poi_adj_c_path = os.path.join(data_path, 'new_poi_adj_matrix_c.npy')
poi_adj_f_path = os.path.join(data_path, 'new_poi_adj_matrix_f.npy')
risk_adj_c_path = os.path.join(data_path, 'new_risk_adj_matrix_c.npy')
risk_adj_f_path = os.path.join(data_path, 'new_risk_adj_matrix_f.npy')
road_adj_c_path = os.path.join(data_path, 'new_road_adj_matrix_c.npy')
road_adj_f_path = os.path.join(data_path, 'new_road_adj_matrix_f.npy')


#映射矩阵：（HW,valid）
grid_node_map_c = np.load(grid_node_map_c_path)
grid_node_map_f = np.load(grid_node_map_f_path)
#静态特征
static_feat_c = np.load(static_feat_c_path)
static_feat_f = np.load(static_feat_f_path)

#语义边
poi_adj_c = np.load(poi_adj_c_path)
poi_adj_f = np.load(poi_adj_f_path)
risk_adj_c = np.load(risk_adj_c_path)
risk_adj_f = np.load(risk_adj_f_path)
road_adj_c = np.load(road_adj_c_path)
road_adj_f = np.load(road_adj_f_path)

# 构造数据集
X_c, y_c, node_X_c, target_time_features= dataloader.dataset_generate(all_data_c_path,grid_node_map_c)
X_f, y_f, node_X_f, _= dataloader.dataset_generate(all_data_f_path,grid_node_map_f)

dataset = TimeSeriesDataset(X_c, y_c, X_f, y_f, node_X_c, node_X_f, target_time_features)

# 设置训练集、验证集和测试集的比例为 6:2:2
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size

# 划分数据集
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 超参数设置
batch_size = 32
num_workers = 0
learning_rate = 0.001
num_epochs = 200

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将常量数据移动到设备上
grid_node_map_c = torch.from_numpy(grid_node_map_c).to(device, dtype=torch.float32)
grid_node_map_f = torch.from_numpy(grid_node_map_f).to(device, dtype=torch.float32)

static_feat_c = torch.from_numpy(static_feat_c).to(device, dtype=torch.float32)
static_feat_f = torch.from_numpy(static_feat_f).to(device, dtype=torch.float32)

#语义边
poi_adj_c = torch.from_numpy(poi_adj_c).to(device, dtype=torch.float32)
poi_adj_f = torch.from_numpy(poi_adj_f).to(device, dtype=torch.float32)
risk_adj_c = torch.from_numpy(risk_adj_c).to(device, dtype=torch.float32)
risk_adj_f = torch.from_numpy(risk_adj_f).to(device, dtype=torch.float32)
road_adj_c = torch.from_numpy(road_adj_c).to(device, dtype=torch.float32)
road_adj_f = torch.from_numpy(road_adj_f).to(device, dtype=torch.float32)

# 生成网格掩码
valid_mask_c = torch.sum(grid_node_map_c, dim=1).view(int(np.sqrt(grid_node_map_c.shape[0])), -1).to(device, dtype=torch.float32)
valid_mask_f = torch.sum(grid_node_map_f, dim=1).view(int(np.sqrt(grid_node_map_f.shape[0])), -1).to(device, dtype=torch.float32)

adj_matrices_c = torch.stack([poi_adj_c, risk_adj_c, road_adj_c], dim=0).to(device, dtype=torch.float32)
adj_matrices_f = torch.stack([poi_adj_f, risk_adj_f, road_adj_f], dim=0).to(device, dtype=torch.float32)

model = ICDM_MambaLSTM(
        c_h=10, c_w=10,
        c_static_feat=static_feat_c,
        c_semantic_mats=adj_matrices_c,
        c_grid_node_map=grid_node_map_c,
        c_valid_mask=valid_mask_c,
        
        # 公共参数
        input_dim=7,
        lstm_hidden=128,
        pre_len=1
    ).to(device)

def print_params_count(model):
    total_params = sum(p.numel() for p in model.parameters())  # 计算总参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数
    
    # 格式化输出
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,}")

print_params_count(model)


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


model_name = model.__class__.__name__
logging.info(f"Model Name: {model_name}\n")



def weighted_huber_loss(pred, target, alpha=1.5, delta=0.5):
    """
    加权Huber损失函数
    Args:
        pred: 预测值，形状为 (S, label, node)
        target: 目标值（仅包含0和1），形状与pred相同
        alpha: 标签1的权重倍数
        delta: Huber损失阈值，控制平方损失与线性损失的切换点
    """
    # 计算预测值与目标值的误差
    error = pred - target
    squared_error = error ** 2
    abs_error = torch.abs(error)
    
    # 计算Huber损失项
    huber_loss = torch.where(abs_error <= delta, 
                             0.5 * squared_error, 
                             delta * (abs_error - 0.5 * delta))
    
    # 生成权重矩阵：标签1的位置权重为alpha，否则为1
    weights = torch.where(target == 1, alpha, 1.0)
    
    # 计算加权损失并取均值
    weighted_loss = (huber_loss * weights).mean()
    
    return weighted_loss


def weighted_mse_loss(pred, target, alpha=1.5):
    """
    加权MSE损失函数
    Args:
        pred: 预测值，形状为 (S, label, node)
        target: 目标值（仅包含0和1），形状与pred相同
        alpha: 标签1的权重倍数
    """
    # 计算平方差
    squared_error = (pred - target) ** 2
    
    # 生成权重矩阵：标签1的位置权重为alpha，否则为1
    weights = torch.where(target == 1, alpha, 1.0)
    
    # 计算加权损失并取均值
    weighted_loss = (squared_error * weights).mean()
    
    return weighted_loss


criterion_c = nn.MSELoss()

optimizer_c = optim.Adam(model.c_parameters(), lr=0.001)
scheduler_c = StepLR(optimizer_c, step_size=1, gamma=0.95)  # 每1个epoch衰减5%


# checkpointname=""
# checkpoint = torch.load(checkpointname)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer_c.load_state_dict(checkpoint['optimizer_c_state_dict'])
# logging.info(f"成功加载: {checkpointname}\n")



# 训练循环
import time
for epoch in range(num_epochs):
    model.train()
    t1=time.time()
    for batch in train_loader:
        # 获取数据
        X_c, y_c, X_f, y_f, node_X_c, node_X_f,target_time_feature = batch
        # 将数据移动到设备(GPU)上
        X_c = X_c.to(device)
        y_c = y_c.to(device)
        target_time_feature = target_time_feature.to(device)
        # 前向传播
        c_pred = model(X_c, target_time_feature)
        
        # 将c_pred和y_c从(S, label, H, W)映射为(S, label, node)
        c_pred_mapped = torch.matmul(c_pred.view(c_pred.shape[0], c_pred.shape[1], -1), grid_node_map_c)
        y_c_mapped = torch.matmul(y_c.view(y_c.shape[0], y_c.shape[1], -1), grid_node_map_c)
        
        # 计算预测损失
        loss_c = weighted_mse_loss(c_pred_mapped, y_c_mapped)

        # loss = loss_f +  loss_c
        # 清零所有梯度
        optimizer_c.zero_grad()
        
        # 计算的梯度
        loss_c.backward()
        
        # 再统一更新参数
        optimizer_c.step()

    scheduler_c.step()
    
    # 在验证集上评估模型
    model.eval()
    t2=time.time()
    print(t2-t1)
    with torch.no_grad():
        val_loss_c = 0
        for batch in val_loader:
            # X_c, y_c, X_f, y_f, node_X_c, node_X_f, target_time_feature = batch
            # 获取数据
            X_c, y_c, X_f, y_f, node_X_c, node_X_f,target_time_feature = batch
            # 将数据移动到设备(GPU)上
            X_c = X_c.to(device)
            y_c = y_c.to(device)
            target_time_feature = target_time_feature.to(device)
            # 前向传播
            c_pred = model(X_c,target_time_feature)
                
            
            c_pred_mapped = torch.matmul(c_pred.view(c_pred.shape[0], c_pred.shape[1], -1), grid_node_map_c)
            y_c_mapped = torch.matmul(y_c.view(y_c.shape[0], y_c.shape[1], -1), grid_node_map_c)
        
            loss_c = criterion_c(c_pred_mapped, y_c_mapped)
            

            loss = loss_c 
            
            val_loss_c += loss_c.item()

        
        
        val_loss_c /= len(val_loader)
        val_loss =val_loss_c
        logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logging.info(f"learning rate_c: {optimizer_c.param_groups[0]['lr']:.6f}")
        logging.info(f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
        logging.info(f"Loss_c: {val_loss_c:.4f}")

    # 在测试集上评估模型
    model.eval()
    # t3=time.time()
    # print(t3-t2)
    with torch.no_grad():
        all_c_preds = []
        all_y_c = []
        t3=time.time()
        
        for batch in test_loader:
            # 获取数据
            X_c, y_c, X_f, y_f, node_X_c, node_X_f,target_time_feature = batch
            # 将数据移动到设备(GPU)上
            X_c = X_c.to(device)
            y_c = y_c.to(device)
            X_f = X_f.to(device)
            y_f = y_f.to(device)
            target_time_feature = target_time_feature.to(device)
            # 前向传播
            c_pred = model(X_c,target_time_feature)
            
            # 维度处理 (batch_size, seq_len, H, W) -> (batch_size*seq_len, H*W)
            batch_size, seq_len = c_pred.shape[0], c_pred.shape[1]
            
            
            # 粗粒度预测映射
            c_pred_flat = c_pred.view(batch_size * seq_len, -1)
            y_c_flat = y_c.view(batch_size * seq_len, -1)
            c_pred_mapped = torch.matmul(c_pred_flat, grid_node_map_c)
            y_c_mapped = torch.matmul(y_c_flat, grid_node_map_c)
            
            # 收集结果
            all_c_preds.append(c_pred_mapped)
            all_y_c.append(y_c_mapped)
        t4=time.time()
        print("Mambapre",t4-t3)
        
        # 合并所有结果
        # (total_samples, nodes)
        c_preds = torch.cat(all_c_preds, dim=0)
        y_c = torch.cat(all_y_c, dim=0)

        # 计算 MSE（RMSE）
        MSE_C = calculate_rmse(c_preds, y_c)

        #calculate_dynamic_map
        MAP_C = calculate_dynamic_map(c_preds, y_c)
        ACC_CL = calculate_acc_at_dynamic_L(c_preds, y_c)

        # 计算分类指标
        metrics_C = calculate_metrics(c_preds, y_c)
        F1_C = metrics_C["F1 Score"]
        ACC_C = metrics_C["Accuracy"]
        AUC_PR_C = metrics_C["AUC-PR"]
        AUC_ROC_C = metrics_C["AUC-ROC"]
        Precision_C = metrics_C["Precision"]  # 新增
        Recall_C = metrics_C["Recall"]        # 新增



        # 修改日志输出
        logging.info(f"RMSE(C): {MSE_C:.4f}")
        logging.info(f"MAP(C): {MAP_C:.4f} ")
        logging.info(f"ACC(CL): {ACC_CL:.4f}")



        logging.info(f"F1 Score(C): {F1_C:.4f}")
        logging.info(f"Accuracy(C): {ACC_C:.4f}")
        logging.info(f"Recall(C): {Recall_C:.4f}")
        logging.info(f"Precision(C): {Precision_C:.4f}") 
        logging.info(f"AUC-PR(C): {AUC_PR_C:.4f}")
        logging.info(f"AUC-ROC(C): {AUC_ROC_C:.4f}")

        # logging.info(f"Acc@15(C): {ACC_C_15*100:.2f}% Acc@20(C): {ACC_C_20*100:.2f}% Acc@25(C): {ACC_C_25*100:.2f}% Acc@30(C): {ACC_C_30*100:.2f}%")
        # logging.info(f"Acc@40(F): {ACC_F_40*100:.2f}% Acc@50(F): {ACC_F_50*100:.2f}% Acc@60(F): {ACC_F_60*100:.2f}% Acc@70(F): {ACC_F_70*100:.2f}%")
        # logging.info(f"MAP@15(C): {MAP_C_15*100:.2f}% MAP@20(C): {MAP_C_20*100:.2f}% MAP@25(C): {MAP_C_25*100:.2f}% MAP@30(C): {MAP_C_30*100:.2f}%")
        # logging.info(f"MAP@40(F): {MAP_F_40*100:.2f}% MAP@50(F): {MAP_F_50*100:.2f}% MAP@60(F): {MAP_F_60*100:.2f}% MAP@70(F): {MAP_F_70*100:.2f}%")
        logging.info("")  # 添加空行保持原格式

    # 在epoch结束后保存权重
    save_dir = "saved_models/"+str(timestamp)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")

    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_c_state_dict': optimizer_c.state_dict(),
        'scheduler_c_state_dict': scheduler_c.state_dict(),
        'loss': loss.item()
    }, checkpoint_path)

    print(f"Saved epoch {epoch+1}")

