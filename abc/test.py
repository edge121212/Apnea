"""
測試腳本：載入訓練好的模型進行測試和預測
"""

import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from model import HybridOSA_Model, load_data
from config import Config

def load_trained_model(model_path, device):
    """
    載入訓練好的模型
    """
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 創建模型實例
    model = HybridOSA_Model()
    
    # 載入檢查點
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 設置為評估模式
    model.eval()
    model.to(device)
    
    print(f"Model loaded successfully!")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Best validation loss: {checkpoint['loss']:.4f}")
    print(f"  - Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
    
    return model

def detailed_evaluation(model, test_loader, device):
    """
    詳細的模型評估
    """
    print("\n" + "="*60)
    print("DETAILED MODEL EVALUATION")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            
            # 獲取預測類別
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # 獲取預測機率
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
    
    # 轉換為 numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 計算各種指標
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    # ROC AUC (使用呼吸中止類別的機率)
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    
    # 打印結果
    print(f"測試樣本數量: {len(all_labels)}")
    print(f"正常樣本: {np.sum(all_labels == 0)} ({np.mean(all_labels == 0)*100:.1f}%)")
    print(f"呼吸中止樣本: {np.sum(all_labels == 1)} ({np.mean(all_labels == 1)*100:.1f}%)")
    print("\n性能指標:")
    print(f"  準確率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  精確率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall):    {recall:.4f}")
    print(f"  F1 分數 (F1-Score): {f1:.4f}")
    print(f"  ROC AUC:           {auc:.4f}")
    
    # 混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n混淆矩陣:")
    print(f"           預測")
    print(f"實際    正常  呼吸中止")
    print(f"正常    {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"呼吸中止  {cm[1,0]:4d}     {cm[1,1]:4d}")
    
    # 詳細分類報告
    print(f"\n詳細分類報告:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['正常', '呼吸中止'],
                              digits=4))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm
    }

def main():
    # 自動尋找 abc 資料夾下的 results/best_model.pth
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    model_path = os.path.join(output_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")

    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 載入模型
    model = load_trained_model(model_path, device)

    # 載入測試數據
    print("\n載入測試數據...")
    _, _, test_loader = load_data(batch_size=Config.BATCH_SIZE)

    if test_loader is None:
        print("錯誤: 無法載入測試數據")
        return

    print(f"測試數據載入成功，共 {len(test_loader)} 個批次")

    # 進行詳細評估
    results = detailed_evaluation(model, test_loader, device)

    print("測試完成!")

if __name__ == "__main__":
    main()
