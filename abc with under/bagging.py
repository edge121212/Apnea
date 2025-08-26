import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from model import HybridOSA_Model, FocalLoss  # 改用 model.py 的 HybridOSA_Model
from config import Config
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, recall_score, precision_score

# ====== 參數設定 ======
N_MODELS = 20
base_dir = Config.SPLIT_DATA_DIR
trainX_path = os.path.join(base_dir, "trainX.npy")
trainY_path = os.path.join(base_dir, "trainY.npy")
valX_path = os.path.join(base_dir, "valX.npy")
valY_path = os.path.join(base_dir, "valY.npy")
testX_path = os.path.join(base_dir, "testX.npy")
testY_path = os.path.join(base_dir, "testY.npy")
output_dir = os.path.join(os.path.dirname(__file__), "ensemble_models")
os.makedirs(output_dir, exist_ok=True)

def undersample(X, Y):
    pos_idx = np.where(Y == 1)[0]
    neg_idx = np.where(Y == 0)[0]
    np.random.seed()
    num_pos = len(pos_idx)
    # 隨機抓 num_pos 個 N
    if len(neg_idx) > num_pos:
        neg_sampled = np.random.choice(neg_idx, num_pos, replace=False)
    else:
        neg_sampled = neg_idx
    final_idx = np.concatenate([pos_idx, neg_sampled])
    np.random.shuffle(final_idx)
    return X[final_idx], Y[final_idx]

# 載入 test 資料
X_test = torch.tensor(np.load(testX_path), dtype=torch.float32)
y_test = np.load(testY_path)
test_loader = DataLoader(TensorDataset(X_test, torch.tensor(y_test, dtype=torch.long)), batch_size=Config.BATCH_SIZE)

all_test_preds = []

for i in range(N_MODELS):
    print(f"\n=== 訓練第 {i+1} 個模型 ===")
    # 載入原始 train/val
    X_train = np.load(trainX_path)
    y_train = np.load(trainY_path)
    X_val = np.load(valX_path)
    y_val = np.load(valY_path)
    # 隨機 undersampling
    X_train_u, y_train_u = undersample(X_train, y_train)
    X_val_u, y_val_u = undersample(X_val, y_val)
    # 轉 tensor
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_u, dtype=torch.float32),
                                            torch.tensor(y_train_u, dtype=torch.long)),
                             batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_u, dtype=torch.float32),
                                          torch.tensor(y_val_u, dtype=torch.long)),
                            batch_size=Config.BATCH_SIZE)
    # 建立模型
    model = HybridOSA_Model()
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    criterion = FocalLoss(gamma=Config.FOCAL_GAMMA)
    # 訓練
    best_f1 = -1
    best_state = None
    patience = 10
    wait = 0
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # 訓練集 performance
        model.eval()
        train_preds = []
        train_labels = []
        train_probs = []
        with torch.no_grad():
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(y.cpu().numpy())
                train_probs.extend(probs.cpu().numpy())
        train_preds = np.array(train_preds)
        train_labels = np.array(train_labels)
        train_probs = np.array(train_probs)
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        acc_tr = accuracy_score(train_labels, train_preds)
        f1_tr = f1_score(train_labels, train_preds, average='binary')
        try:
            auc_tr = roc_auc_score(train_labels, train_probs)
        except:
            auc_tr = float('nan')
        print(f"Model {i+1} train — Epoch {epoch+1} — Acc: {acc_tr:.4f}, F1: {f1_tr:.4f}, AUROC: {auc_tr:.4f}")

        # 驗證集 performance
        val_preds = []
        val_labels = []
        val_probs = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_probs = np.array(val_probs)
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average='binary')
        try:
            auc = roc_auc_score(val_labels, val_probs)
        except:
            auc = float('nan')
        print(f"Model {i+1} validation — Epoch {epoch+1} — Acc: {acc:.4f}, F1: {f1:.4f}, AUROC: {auc:.4f}")

        # early stopping 以 val F1 為指標
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1} (no val F1 improvement for {patience} epochs)")
            break

    # 用最佳模型權重
    if best_state is not None:
        model.load_state_dict(best_state)

    # 儲存模型
    torch.save(model.state_dict(), os.path.join(output_dir, f"model_{i+1}.pth"))
    print(f"模型已儲存：model_{i+1}.pth")
    # 測試
    test_preds = []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
    all_test_preds.append(test_preds)

# 多數決 ensemble

# ====== 測試集多數決 ======
all_test_preds = np.array(all_test_preds)  # shape: [N_MODELS, num_test]
majority_vote_test = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_test_preds)
# 取得所有模型的預測機率
all_test_probs = []
for i in range(N_MODELS):
    X_test = torch.tensor(np.load(testX_path), dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(X_test, torch.tensor(y_test, dtype=torch.long)), batch_size=Config.BATCH_SIZE)
    model = HybridOSA_Model()
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(os.path.join(output_dir, f"model_{i+1}.pth"), map_location=device))
    model.to(device)
    model.eval()
    test_probs = []
    with torch.no_grad():
        for X, _ in test_loader:
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
    all_test_probs.append(test_probs)
all_test_probs = np.array(all_test_probs)  # shape: [N_MODELS, num_test]
mean_test_probs = np.mean(all_test_probs, axis=0)
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, recall_score, precision_score
# 測試集 performance
acc_test = accuracy_score(y_test, majority_vote_test)
f1_test = f1_score(y_test, majority_vote_test, average='binary')
recall_test = recall_score(y_test, majority_vote_test, average='binary')
precision_test = precision_score(y_test, majority_vote_test, average='binary')
try:
    auc_test = roc_auc_score(y_test, mean_test_probs)
except:
    auc_test = float('nan')
cm_test = confusion_matrix(y_test, majority_vote_test)
print(f"\nEnsemble 多數決測試集準確率: {acc_test:.4f}")
print(f"Ensemble 多數決測試集 F1: {f1_test:.4f}")
print(f"Ensemble 多數決測試集 Recall: {recall_test:.4f}")
print(f"Ensemble 多數決測試集 Precision: {precision_test:.4f}")
print(f"Ensemble 多數決測試集 AUROC: {auc_test:.4f}")
print(f"Ensemble 多數決測試集混淆矩陣:")
print(cm_test)
print("\n詳細分類報告:")
print(classification_report(y_test, majority_vote_test, target_names=['正常', '呼吸中止'], digits=4))

# ====== 驗證集多數決 ======
all_val_preds = []
all_val_probs = []
for i in range(N_MODELS):
    X_val = np.load(valX_path)
    y_val = np.load(valY_path)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.long)),
                            batch_size=Config.BATCH_SIZE)
    model = HybridOSA_Model()
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(os.path.join(output_dir, f"model_{i+1}.pth"), map_location=device))
    model.to(device)
    model.eval()
    val_preds = []
    val_probs = []
    with torch.no_grad():
        for X, _ in val_loader:
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())
    all_val_preds.append(val_preds)
    all_val_probs.append(val_probs)

all_val_preds = np.array(all_val_preds)  # shape: [N_MODELS, num_val]
all_val_probs = np.array(all_val_probs)  # shape: [N_MODELS, num_val]
majority_vote_val = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_val_preds)
mean_val_probs = np.mean(all_val_probs, axis=0)
acc_val = accuracy_score(y_val, majority_vote_val)
f1_val = f1_score(y_val, majority_vote_val, average='binary')
recall_val = recall_score(y_val, majority_vote_val, average='binary')
precision_val = precision_score(y_val, majority_vote_val, average='binary')
try:
    auc_val = roc_auc_score(y_val, mean_val_probs)
except:
    auc_val = float('nan')
cm_val = confusion_matrix(y_val, majority_vote_val)
print(f"\nEnsemble 多數決驗證集準確率: {acc_val:.4f}")
print(f"Ensemble 多數決驗證集 F1: {f1_val:.4f}")
print(f"Ensemble 多數決驗證集 Recall: {recall_val:.4f}")
print(f"Ensemble 多數決驗證集 Precision: {precision_val:.4f}")
print(f"Ensemble 多數決驗證集 AUROC: {auc_val:.4f}")
print(f"Ensemble 多數決驗證集混淆矩陣:")
print(cm_val)
print("\n詳細分類報告:")
print(classification_report(y_val, majority_vote_val, target_names=['正常', '呼吸中止'], digits=4))