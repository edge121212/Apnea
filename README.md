# Apnea

本專案為睡眠呼吸中止症（Apnea）相關的資料處理與模型訓練/測試工具。

## 環境需求

- Python 3.8 以上
- 建議使用虛擬環境（venv 或 conda）

## 安裝套件

```bash
pip install -r requirements.txt
```
（請自行建立 requirements.txt 或補充所需套件）

## 執行流程

### 1. 資料前處理
執行 `abc/data_process.py` 進行資料分割與前處理。
> 注意：請根據你的資料路徑，修改 `config.py` 內的檔案路徑設定（如原始資料、splitdata 位置等）。

```bash
python abc/data_process.py
```
處理後的分割資料會依照 `config.py` 設定存放。

### 2. 模型訓練
執行 `abc/train.py` 進行模型訓練。
> 注意：訓練參數、資料路徑等都需在 `config.py` 內自行調整。

```bash
python abc/train.py
```
訓練完成後，模型與訓練紀錄會存放在 `abc/results/` 目錄下（如 best_model.pth、training_history.png 等）。

### 3. 模型測試
執行 `abc/test.py` 進行模型測試。
> 注意：測試資料路徑、模型路徑等同樣需在 `config.py` 內設定。

```bash
python abc/test.py
```
測試結果會存放在 `abc/results/` 目錄下（如 training_results.json）。

---

## 其他說明

- 若有 ensemble 或 bagging 相關功能，請參考 `abc with under/` 或 `abc+x with under/` 目錄下的腳本。
- 每個資料夾（如 abc, abc+x）都需根據自身需求修改 `config.py`，並分別執行 train/test。
