"""
配置文件：包含模型訓練和數據處理的所有超參數
"""

class Config:
    # 數據路徑配置
    BASE_DIR = "put your base dir here"
    SPLIT_DATA_DIR = "put your split data dir here"

    # 數據預處理參數
    SAMPLING_RATE = 100  # Hz
    WINDOW_LENGTH = 6    # minutes
    OUTPUT_LENGTH = 1024 # resampled length
    TRAIN_VAL_SPLIT = 0.9
    
    # 小波去噪參數
    WAVELET = "coif4"
    WAVELET_LEVELS = 6
    
    # 心率過濾範圍
    MIN_HR = 40  # bpm
    MAX_HR = 180 # bpm
    
    # 模型架構參數
    INPUT_CHANNELS = 4
    MPCA_CHANNELS = [32, 64]
    SE_REDUCTION = 16
    
    # Transformer參數
    D_MODEL = 64
    N_HEADS = 4
    N_LAYERS = 3
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1
    
    # 訓練參數
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 30
    
    # Focal Loss參數
    FOCAL_GAMMA = 2
    FOCAL_ALPHA = None
    
    # 其他
    NUM_CLASSES = 2
    DEVICE = "cuda"  # or "cpu"
