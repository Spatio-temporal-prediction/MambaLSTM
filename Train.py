import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
import os
import sys
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
import logging
import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
from main import Main_MambaLSTM
from data_process import dataloader

torch.autograd.set_detect_anomaly(True)

def log_init():
    log_dir = "log_ICDM"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_{timestamp}.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

log_init()

def evaluate_single_step(pred, true, L=None):
    device = pred.device
    batch_size, num_nodes = pred.shape

    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()

    metrics = {}

    rmse_error = np.mean((pred_np - true_np) ** 2)
    metrics['RMSE'] = np.sqrt(rmse_error)

    if L is not None:
        acc_list = []
        for b in range(batch_size):
            pred_top = np.argsort(-pred_np[b])[:L]
            true_top = np.argsort(-true_np[b])[:L]
            intersection = np.intersect1d(pred_top, true_top)
            acc_list.append(len(intersection) / L)
        metrics[f'Acc@{L}'] = np.mean(acc_list)

    map_list = []
    for b in range(batch_size):
        y_true = true_np[b]
        y_score = pred_np[b]
        if np.unique(y_true).size > 1:
            ap = average_precision_score(y_true, y_score)
        else:
            ap = 0.0
        map_list.append(ap)

    metrics['MAP'] = np.mean(map_list)
    return metrics

def calculate_metrics(pred, true):
    pred_np = pred.detach().cpu().numpy().flatten()
    true_np = true.detach().cpu().numpy().flatten()

    metrics = {}

    try:
        metrics["AUC-ROC"] = roc_auc_score(true_np, pred_np)
    except ValueError:
        metrics["AUC-ROC"] = float('nan')

    metrics["AUC-PR"] = average_precision_score(true_np, pred_np)

    pred_binary = (pred_np >= 0.5).astype(np.int32)
    
    metrics["Accuracy"] = accuracy_score(true_np, pred_binary)
    metrics["Precision"] = precision_score(true_np, pred_binary, zero_division=0)
    metrics["Recall"] = recall_score(true_np, pred_binary, zero_division=0)
    metrics["F1 Score"] = f1_score(true_np, pred_binary, zero_division=0)

    return metrics

def calculate_rmse(pred, true):
    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()
    return np.sqrt(np.mean((pred_np - true_np) ** 2))

def calculate_acc_at_L(pred, true, L):
    batch_size, num_nodes = pred.shape
    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()
    
    acc_list = []
    for b in range(batch_size):
        pred_top = np.argsort(-pred_np[b])[:L]
        true_top = np.argsort(-true_np[b])[:L]
        intersection = np.intersect1d(pred_top, true_top)
        acc_list.append(len(intersection) / L)
    return np.mean(acc_list)

def calculate_acc_at_dynamic_L(pred, true):
    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()
    batch_size, num_nodes = pred_np.shape
    
    acc_list = []
    for b in range(batch_size):
        true_labels = true_np[b]
        L = int(np.sum(true_labels))
        if L == 0:
            acc_list.append(0.0)
            continue
        
        pred_top = np.argsort(-pred_np[b])[:L]
        true_top = np.where(true_labels == 1)[0]
        intersection = np.intersect1d(pred_top, true_top)
        acc_list.append(len(intersection) / L)
    
    return np.mean(acc_list) if acc_list else 0.0

def calculate_dynamic_map(pred, true):
    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()
    batch_size, num_nodes = pred_np.shape
    
    map_list = []
    for b in range(batch_size):
        pred_scores = pred_np[b]
        true_labels = true_np[b]
        L = int(np.sum(true_labels))
        if L == 0:
            continue
        
        pred_top = np.argsort(-pred_scores)[:L]
        true_positives = np.where(true_labels == 1)[0]
        true_set = set(true_positives)
        
        cumulative_correct = 0
        precision_sum = 0.0
        
        for rank, node in enumerate(pred_top, 1):
            hit = 1 if node in true_set else 0
            cumulative_correct += hit
            current_precision = cumulative_correct / rank
            precision_sum += current_precision * hit
        
        ap = precision_sum / L if L > 0 else 0.0
        map_list.append(ap)
    
    return np.mean(map_list) if map_list else 0.0

def calculate_map(pred, true, L):
    pred_np = pred.cpu().numpy()
    true_np = true.cpu().numpy()
    batch_size, num_nodes = pred_np.shape
    
    map_list = []
    
    for b in range(batch_size):
        pred_scores = pred_np[b]
        true_scores = true_np[b]
        pred_topL = np.argsort(-pred_scores)[:L]
        true_topL = np.argsort(-true_scores)[:L]
        true_set = set(true_topL)
        
        cumulative_correct = 0
        sum_precision = 0.0
        
        for position in range(L):
            current_node = pred_topL[position]
            real = 1 if current_node in true_set else 0
            cumulative_correct += real
            denominator = position + 1
            precision_at_j = cumulative_correct / denominator if denominator !=0 else 0.0
            sum_precision += precision_at_j * real
        
        ap = sum_precision / L
        map_list.append(ap)
    
    return np.mean(map_list)

class TimeSeriesDataset(Dataset):
    def __init__(self, grid_X_c, grid_y_c, grid_X_f, grid_y_f, node_X_c, node_X_f, target_time_feature):
        self.X_c = torch.tensor(grid_X_c, dtype=torch.float32)
        self.y_c = torch.tensor(grid_y_c, dtype=torch.float32)
        self.X_f = torch.tensor(grid_X_f, dtype=torch.float32)
        self.y_f = torch.tensor(grid_y_f, dtype=torch.float32)
        self.node_X_c = torch.tensor(node_X_c, dtype=torch.float32)
        self.node_X_f = torch.tensor(node_X_f, dtype=torch.float32)
        self.target_time_feature = torch.tensor(target_time_feature, dtype=torch.float32)

    def __len__(self):
        return len(self.X_c)

    def __getitem__(self, idx):
        return self.X_c[idx], self.y_c[idx], self.X_f[idx], self.y_f[idx], self.node_X_c[idx], self.node_X_f[idx], self.target_time_feature[idx]

data_path = os.path.join(os.path.split(curPath)[0], "data", 'npy_new_nodiff')

all_data_c_path = os.path.join(data_path, 'new_grid_data_c_4d.npy')
all_data_f_path = os.path.join(data_path, 'new_grid_data_f_4d.npy')

grid_node_map_c_path = os.path.join(data_path, 'new_grid_node_map_c.npy')
grid_node_map_f_path = os.path.join(data_path, 'new_grid_node_map_f.npy')

static_feat_c_path = os.path.join(data_path, 'new_static_feat_c.npy')
static_feat_f_path = os.path.join(data_path, 'new_static_feat_f.npy')

poi_adj_c_path = os.path.join(data_path, 'new_poi_adj_matrix_c.npy')
poi_adj_f_path = os.path.join(data_path, 'new_poi_adj_matrix_f.npy')
risk_adj_c_path = os.path.join(data_path, 'new_risk_adj_matrix_c.npy')
risk_adj_f_path = os.path.join(data_path, 'new_risk_adj_matrix_f.npy')
road_adj_c_path = os.path.join(data_path, 'new_road_adj_matrix_c.npy')
road_adj_f_path = os.path.join(data_path, 'new_road_adj_matrix_f.npy')

grid_node_map_c = np.load(grid_node_map_c_path)
grid_node_map_f = np.load(grid_node_map_f_path)
static_feat_c = np.load(static_feat_c_path)
static_feat_f = np.load(static_feat_f_path)

poi_adj_c = np.load(poi_adj_c_path)
poi_adj_f = np.load(poi_adj_f_path)
risk_adj_c = np.load(risk_adj_c_path)
risk_adj_f = np.load(risk_adj_f_path)
road_adj_c = np.load(road_adj_c_path)
road_adj_f = np.load(road_adj_f_path)

X_c, y_c, node_X_c, target_time_features = dataloader.dataset_generate(all_data_c_path, grid_node_map_c)
X_f, y_f, node_X_f, _ = dataloader.dataset_generate(all_data_f_path, grid_node_map_f)

dataset = TimeSeriesDataset(X_c, y_c, X_f, y_f, node_X_c, node_X_f, target_time_features)

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

batch_size = 32
num_workers = 0
learning_rate = 0.001
num_epochs = 200

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

grid_node_map_c = torch.from_numpy(grid_node_map_c).to(device, dtype=torch.float32)
grid_node_map_f = torch.from_numpy(grid_node_map_f).to(device, dtype=torch.float32)

static_feat_c = torch.from_numpy(static_feat_c).to(device, dtype=torch.float32)
static_feat_f = torch.from_numpy(static_feat_f).to(device, dtype=torch.float32)

poi_adj_c = torch.from_numpy(poi_adj_c).to(device, dtype=torch.float32)
poi_adj_f = torch.from_numpy(poi_adj_f).to(device, dtype=torch.float32)
risk_adj_c = torch.from_numpy(risk_adj_c).to(device, dtype=torch.float32)
risk_adj_f = torch.from_numpy(risk_adj_f).to(device, dtype=torch.float32)
road_adj_c = torch.from_numpy(road_adj_c).to(device, dtype=torch.float32)
road_adj_f = torch.from_numpy(road_adj_f).to(device, dtype=torch.float32)

valid_mask_c = torch.sum(grid_node_map_c, dim=1).view(int(np.sqrt(grid_node_map_c.shape[0])), -1).to(device, dtype=torch.float32)
valid_mask_f = torch.sum(grid_node_map_f, dim=1).view(int(np.sqrt(grid_node_map_f.shape[0])), -1).to(device, dtype=torch.float32)

adj_matrices_c = torch.stack([poi_adj_c, risk_adj_c, road_adj_c], dim=0).to(device, dtype=torch.float32)
adj_matrices_f = torch.stack([poi_adj_f, risk_adj_f, road_adj_f], dim=0).to(device, dtype=torch.float32)

model = Main_MambaLSTM(
    c_h=10,
    c_w=10,
    c_static_feat=static_feat_c,
    c_semantic_mats=adj_matrices_c,
    c_grid_node_map=grid_node_map_c,
    c_valid_mask=valid_mask_c,
    input_dim=7,
    lstm_hidden=128,
    pre_len=1
).to(device)

def print_params_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,}")

print_params_count(model)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = model.__class__.__name__
logging.info(f"Model Name: {model_name}\n")

def weighted_huber_loss(pred, target, alpha=1.5, delta=0.5):
    error = pred - target
    squared_error = error ** 2
    abs_error = torch.abs(error)
    
    huber_loss = torch.where(abs_error <= delta, 
                            0.5 * squared_error, 
                            delta * (abs_error - 0.5 * delta))
    
    weights = torch.where(target == 1, alpha, 1.0)
    return (huber_loss * weights).mean()

def weighted_mse_loss(pred, target, alpha=1.5):
    squared_error = (pred - target) ** 2
    weights = torch.where(target == 1, alpha, 1.0)
    return (squared_error * weights).mean()

criterion_c = nn.MSELoss()
optimizer_c = optim.Adam(model.c_parameters(), lr=0.001)
scheduler_c = StepLR(optimizer_c, step_size=1, gamma=0.95)

for epoch in range(num_epochs):
    model.train()
    t1 = time.time()
    for batch in train_loader:
        X_c, y_c, X_f, y_f, node_X_c, node_X_f, target_time_feature = batch
        X_c = X_c.to(device)
        y_c = y_c.to(device)
        target_time_feature = target_time_feature.to(device)
        
        c_pred = model(X_c, target_time_feature)
        c_pred_mapped = torch.matmul(c_pred.view(c_pred.shape[0], c_pred.shape[1], -1), grid_node_map_c)
        y_c_mapped = torch.matmul(y_c.view(y_c.shape[0], y_c.shape[1], -1), grid_node_map_c)
        loss_c = weighted_mse_loss(c_pred_mapped, y_c_mapped)

        optimizer_c.zero_grad()
        loss_c.backward()
        optimizer_c.step()

    scheduler_c.step()
    
    model.eval()
    t2 = time.time()
    print(t2 - t1)
    with torch.no_grad():
        val_loss_c = 0
        for batch in val_loader:
            X_c, y_c, X_f, y_f, node_X_c, node_X_f, target_time_feature = batch
            X_c = X_c.to(device)
            y_c = y_c.to(device)
            target_time_feature = target_time_feature.to(device)
            
            c_pred = model(X_c, target_time_feature)
            c_pred_mapped = torch.matmul(c_pred.view(c_pred.shape[0], c_pred.shape[1], -1), grid_node_map_c)
            y_c_mapped = torch.matmul(y_c.view(y_c.shape[0], y_c.shape[1], -1), grid_node_map_c)
            loss_c = criterion_c(c_pred_mapped, y_c_mapped)
            
            val_loss_c += loss_c.item()

        val_loss_c /= len(val_loader)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
        logging.info(f"learning rate_c: {optimizer_c.param_groups[0]['lr']:.6f}")
        logging.info(f"Train Loss: {loss_c.item():.4f}, Val Loss: {val_loss_c:.4f}")

    model.eval()
    with torch.no_grad():
        all_c_preds = []
        all_y_c = []
        t3 = time.time()
        
        for batch in test_loader:
            X_c, y_c, X_f, y_f, node_X_c, node_X_f, target_time_feature = batch
            X_c = X_c.to(device)
            y_c = y_c.to(device)
            X_f = X_f.to(device)
            y_f = y_f.to(device)
            target_time_feature = target_time_feature.to(device)
            
            c_pred = model(X_c, target_time_feature)
            
            batch_size, seq_len = c_pred.shape[0], c_pred.shape[1]
            c_pred_flat = c_pred.view(batch_size * seq_len, -1)
            y_c_flat = y_c.view(batch_size * seq_len, -1)
            c_pred_mapped = torch.matmul(c_pred_flat, grid_node_map_c)
            y_c_mapped = torch.matmul(y_c_flat, grid_node_map_c)
            
            all_c_preds.append(c_pred_mapped)
            all_y_c.append(y_c_mapped)
        
        t4 = time.time()
        print("Mambapre", t4 - t3)
        
        c_preds = torch.cat(all_c_preds, dim=0)
        y_c = torch.cat(all_y_c, dim=0)

        MSE_C = calculate_rmse(c_preds, y_c)
        MAP_C = calculate_dynamic_map(c_preds, y_c)
        ACC_CL = calculate_acc_at_dynamic_L(c_preds, y_c)

        metrics_C = calculate_metrics(c_preds, y_c)
        F1_C = metrics_C["F1 Score"]
        ACC_C = metrics_C["Accuracy"]
        AUC_PR_C = metrics_C["AUC-PR"]
        AUC_ROC_C = metrics_C["AUC-ROC"]
        Precision_C = metrics_C["Precision"]
        Recall_C = metrics_C["Recall"]

        logging.info(f"RMSE(C): {MSE_C:.4f}")
        logging.info(f"MAP(C): {MAP_C:.4f} ")
        logging.info(f"ACC(CL): {ACC_CL:.4f}")
        logging.info(f"F1 Score(C): {F1_C:.4f}")
        logging.info(f"Accuracy(C): {ACC_C:.4f}")
        logging.info(f"Recall(C): {Recall_C:.4f}")
        logging.info(f"Precision(C): {Precision_C:.4f}") 
        logging.info(f"AUC-PR(C): {AUC_PR_C:.4f}")
        logging.info(f"AUC-ROC(C): {AUC_ROC_C:.4f}")
        logging.info("")

    save_dir = f"saved_models/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_c_state_dict': optimizer_c.state_dict(),
        'scheduler_c_state_dict': scheduler_c.state_dict(),
        'loss': loss_c.item()
    }, checkpoint_path)
    print(f"Saved epoch {epoch+1}")
