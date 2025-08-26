"""
工具函數：包含模型保存、載入和結果可視化等功能
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import json
from datetime import datetime

def save_model(model, optimizer, epoch, loss, filepath):
    """保存模型檢查點"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(model, optimizer, filepath):
    """載入模型檢查點"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {filepath}, epoch: {epoch}, loss: {loss:.4f}")
    return epoch, loss

def plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Apnea'], save_path=None):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_scores, save_path=None):
    """繪製 ROC 曲線"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(train_losses, val_losses=None, save_path=None):
    """繪製訓練過程中的損失曲線"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_results(results, filepath):
    """保存評估結果到 JSON 文件"""
    results['timestamp'] = datetime.now().isoformat()
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filepath}")

def print_model_summary(model):
    """
    打印模型總結信息 (不進行前向傳播測試)
    """
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    
    # 計算參數數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 移除前向傳播測試部分
    # 不再進行 dummy input 測試
    
    print("=" * 50)

def ensure_dir(directory):
    """確保目錄存在，如果不存在則創建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def get_class_distribution(labels):
    """獲取類別分布統計"""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print("Class Distribution:")
    for cls, count in zip(unique, counts):
        percentage = count / total * 100
        class_name = "Normal" if cls == 0 else "Apnea"
        print(f"  {class_name} (Class {cls}): {count} samples ({percentage:.2f}%)")
    
    return dict(zip(unique, counts))
