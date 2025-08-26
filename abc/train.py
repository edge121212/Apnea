"""
訓練腳本：提供完整的訓練流程，包含模型保存和結果可視化
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from model import HybridOSA_Model, load_data, FocalLoss, calculate_class_weights_auto, count_params
from utils import save_model, save_results, print_model_summary, plot_training_history
from config import Config

def evaluate_model(model, dataloader, device, name="Test"):
    """
    評估模型性能
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary')
    rec = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = float('nan')  # 若無法算 AUROC (如只有一類)

    print(f"{name} — Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUROC: {auc:.4f}")
    return acc, prec, rec, f1, auc

def main():
    # 創建輸出目錄於 abc 資料夾
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(batch_size=Config.BATCH_SIZE)
    
    print("Initializing model...")
    model = HybridOSA_Model()
    print_model_summary(model)
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # 設置優化器和損失函數
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 計算類別權重來處理不平衡問題
    print("Calculating class weights...")
    class_weights = calculate_class_weights_auto(train_loader, device)
    
    # 使用類別權重的 Focal Loss
    criterion = FocalLoss(gamma=Config.FOCAL_GAMMA, alpha=class_weights)
    
    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs...")
    
    # 存儲訓練歷史
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        # 訓練
        model.train()
        train_loss = 0
        
        # 使用 tqdm 顯示訓練進度
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for X, y in train_pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # 更新進度條顯示當前批次損失
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 驗證
        model.eval()
        val_loss = 0
        
        # 使用 tqdm 顯示驗證進度
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        with torch.no_grad():
            for X, y in val_pbar:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()
                
                # 更新進度條顯示當前批次損失
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, optimizer, epoch, avg_val_loss, 
                      os.path.join(output_dir, "best_model.pth"))
            print(f"  ✓ Saved new best model (loss: {avg_val_loss:.4f})")
        
        # 評估
        print("  Validation Results:")
        evaluate_model(model, val_loader, device, name="Validation")
        print("-" * 50)
    
    # 繪製訓練曲線
    plot_training_history(train_losses, val_losses, 
                         save_path=os.path.join(output_dir, "training_history.png"))
    
    # 保存最終模型
    save_model(model, optimizer, Config.NUM_EPOCHS-1, avg_val_loss,
              os.path.join(output_dir, "final_model.pth"))
    
    # 保存配置和結果
    results = {
        "model_config": {
            "architecture": "HybridOSA_Model (MPCA + Transformer)",
            "input_channels": Config.INPUT_CHANNELS,
            "output_length": Config.OUTPUT_LENGTH,
            "d_model": Config.D_MODEL,
            "n_heads": Config.N_HEADS,
            "n_layers": Config.N_LAYERS
        },
        "training_config": {
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.LEARNING_RATE,
            "num_epochs": Config.NUM_EPOCHS,
            "focal_gamma": Config.FOCAL_GAMMA
        },
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val_loss
    }
    
    save_results(results, os.path.join(output_dir, "training_results.json"))
    
    print(f"\nAll results saved in '{output_dir}' directory")
    print("Training completed!")

if __name__ == "__main__":
    main()
