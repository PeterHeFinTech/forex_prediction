# import os
# import numpy as np
# import torch
# import torch.multiprocessing as mp
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler
# import psutil


# if mp.get_start_method(allow_none=True) != 'fork':
#     try:
#         mp.set_start_method('fork', force=True)
#         print("Multiprocessing start method set to 'fork'", flush=True)
#     except RuntimeError:
#         print("Warning: Unable to set multiprocessing start method to 'fork'", flush=True)


# def report_memory(prefix=""):
#     mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
#     print(f"{prefix}Current RAM usage: {mem:.2f} GB", flush=True)


# class ForexClassificationDataset(torch.utils.data.Dataset):
#     def __init__(self, inputs, labels, indices):
#         self.x = inputs
#         self.y = labels
#         self.indices = indices

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         i = self.indices[idx]
#         return self.x[i], self.y[i]


# def create_dataset(data_path, dtype, rank):
#     """
#     从 npz 文件加载数据
#     npz 文件包含：
#     - X_train: (N_train, 96, 4)
#     - y_train: (N_train,)
#     - X_val: (N_val, 96, 4)
#     - y_val: (N_val,)
#     - X_test: (N_test, 96, 4) [可选]
#     - y_test: (N_test,) [可选]
#     """
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
#     # 加载 npz 文件
#     raw_data = np.load(data_path)
    
#     if rank == 0:
#         print(f"Loading data from: {data_path}", flush=True)
#         print(f"Available keys: {raw_data.files}", flush=True)
    
#     # 加载训练集
#     train_inputs = torch.from_numpy(raw_data['X_train']).to(dtype)  # [N_train, 96, 4]
#     train_labels = torch.from_numpy(raw_data['y_train']).long()      # [N_train]
    
#     # 加载验证集
#     val_inputs = torch.from_numpy(raw_data['X_val']).to(dtype)      # [N_val, 96, 4]
#     val_labels = torch.from_numpy(raw_data['y_val']).long()          # [N_val]
    
#     train_size = train_inputs.shape[0]
#     val_size = val_inputs.shape[0]
    
#     if rank == 0:
#         print(f"\nDataset loaded:", flush=True)
#         print(f"  Train - Input shape: {train_inputs.shape}, Label shape: {train_labels.shape}", flush=True)
#         print(f"  Val   - Input shape: {val_inputs.shape}, Label shape: {val_labels.shape}", flush=True)
#         print(f"  Train samples: {train_size:,}, Val samples: {val_size:,}", flush=True)
        
#         # 统计标签分布
#         train_label_counts = torch.bincount(train_labels, minlength=3)
#         val_label_counts = torch.bincount(val_labels, minlength=3)
#         print(f"\nLabel distribution:", flush=True)
#         print(f"  Train - Class 0: {train_label_counts[0]:,} ({100*train_label_counts[0]/train_size:.2f}%), "
#               f"Class 1: {train_label_counts[1]:,} ({100*train_label_counts[1]/train_size:.2f}%), "
#               f"Class 2: {train_label_counts[2]:,} ({100*train_label_counts[2]/train_size:.2f}%)", flush=True)
#         print(f"  Val   - Class 0: {val_label_counts[0]:,} ({100*val_label_counts[0]/val_size:.2f}%), "
#               f"Class 1: {val_label_counts[1]:,} ({100*val_label_counts[1]/val_size:.2f}%), "
#               f"Class 2: {val_label_counts[2]:,} ({100*val_label_counts[2]/val_size:.2f}%)", flush=True)
        
#         # 显示数据统计信息
#         print(f"\nData statistics:", flush=True)
#         print(f"  Train - Min: {train_inputs.min():.6f}, Max: {train_inputs.max():.6f}, "
#               f"Mean: {train_inputs.mean():.6f}, Std: {train_inputs.std():.6f}", flush=True)
#         print(f"  Val   - Min: {val_inputs.min():.6f}, Max: {val_inputs.max():.6f}, "
#               f"Mean: {val_inputs.mean():.6f}, Std: {val_inputs.std():.6f}", flush=True)
    
#     # 创建索引
#     train_indices = np.arange(train_size)
#     val_indices = np.arange(val_size)
    
#     # 创建 Dataset
#     train_dataset = ForexClassificationDataset(train_inputs, train_labels, train_indices)
#     val_dataset = ForexClassificationDataset(val_inputs, val_labels, val_indices)
    
#     if rank == 0:
#         report_memory("After dataset creation: ")
    
#     return train_dataset, val_dataset, train_size, val_size


# def create_test_dataset(data_path, dtype, rank):
#     """
#     单独加载测试集（如果需要的话）
#     """
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
#     raw_data = np.load(data_path)
    
#     if 'X_test' not in raw_data or 'y_test' not in raw_data:
#         if rank == 0:
#             print("Warning: Test set not found in the npz file", flush=True)
#         return None, None, 0
    
#     test_inputs = torch.from_numpy(raw_data['X_test']).to(dtype)
#     test_labels = torch.from_numpy(raw_data['y_test']).long()
    
#     test_size = test_inputs.shape[0]
    
#     if rank == 0:
#         print(f"\nTest set loaded:", flush=True)
#         print(f"  Test - Input shape: {test_inputs.shape}, Label shape: {test_labels.shape}", flush=True)
#         print(f"  Test samples: {test_size:,}", flush=True)
        
#         test_label_counts = torch.bincount(test_labels, minlength=3)
#         print(f"  Test  - Class 0: {test_label_counts[0]:,} ({100*test_label_counts[0]/test_size:.2f}%), "
#               f"Class 1: {test_label_counts[1]:,} ({100*test_label_counts[1]/test_size:.2f}%), "
#               f"Class 2: {test_label_counts[2]:,} ({100*test_label_counts[2]/test_size:.2f}%)", flush=True)
    
#     test_indices = np.arange(test_size)
#     test_dataset = ForexClassificationDataset(test_inputs, test_labels, test_indices)
    
#     return test_dataset, test_size


# def create_dataloader(dataset, batch_size, num_workers, world_size, rank, shuffle=True):
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
#     per_gpu_workers = max(1, num_workers // world_size)
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         num_workers=per_gpu_workers,
#         pin_memory=True,
#         persistent_workers=True,
#         prefetch_factor=2,
#         drop_last=True
#     )
#     return loader, sampler



import os
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import psutil

if mp.get_start_method(allow_none=True) != 'fork':
    try:
        mp.set_start_method('fork', force=True)
        print("Multiprocessing start method set to 'fork'", flush=True)
    except RuntimeError:
        print("Warning: Unable to set multiprocessing start method to 'fork'", flush=True)

def report_memory(prefix=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"{prefix}Current RAM usage: {mem:.2f} GB", flush=True)

class ForexClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, targets, indices):
        self.x = inputs
        self.y = labels
        self.targets = targets  # 第129天的OHLC价格
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 通过重采样后的索引列表来访问原始数据
        # 这样即使数据被多次采样，也不需要复制物理内存中的 Tensor
        i = self.indices[idx]
        return self.x[i], self.y[i], self.targets[i]

def create_dataset(data_path, dtype, rank, balance_train=True):
    """
    参数:
    balance_train (bool): 是否对训练集进行类别平衡（过采样）。默认为 True。
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    # 加载 npz 文件
    raw_data = np.load(data_path)
    
    if rank == 0:
        print(f"Loading data from: {data_path}", flush=True)
    
    # 加载数据到 Tensor
    train_inputs = torch.from_numpy(raw_data['X_train']).to(dtype)
    train_labels = torch.from_numpy(raw_data['y_train']).long()
    train_targets = torch.from_numpy(raw_data['targets_train']).to(dtype)  # 第129天价格
    
    val_inputs = torch.from_numpy(raw_data['X_val']).to(dtype)
    val_labels = torch.from_numpy(raw_data['y_val']).long()
    val_targets = torch.from_numpy(raw_data['targets_val']).to(dtype)
    
    train_size_original = train_inputs.shape[0]
    val_size = val_inputs.shape[0]

    # --- 核心修改：生成训练集索引（支持类别平衡）---
    if balance_train:
        # 1. 统计各类别及其索引
        # 注意：这里假设 labels 是 CPU tensor，如果很大且在 GPU，需转回 CPU 处理索引
        y_np = train_labels.numpy()
        indices_0 = np.where(y_np == 0)[0]
        indices_1 = np.where(y_np == 1)[0]
        indices_2 = np.where(y_np == 2)[0]

        count_0 = len(indices_0)
        count_1 = len(indices_1)
        count_2 = len(indices_2)

        # 2. 确定目标数量（以最大类别的数量为基准，进行过采样）
        max_count = max(count_0, count_1, count_2)
        
        if rank == 0:
            print(f"\n[Balancing Strategy] Original counts -> 0: {count_0}, 1: {count_1}, 2: {count_2}", flush=True)
            print(f"[Balancing Strategy] Upsampling minority classes to: {max_count}", flush=True)

        # 3. 随机重采样 (Resampling with replacement)
        # 这里的 replace=True 允许重复抽取，从而增加少数类的样本数
        balanced_idx_0 = np.random.choice(indices_0, max_count, replace=True)
        balanced_idx_1 = np.random.choice(indices_1, max_count, replace=True)
        balanced_idx_2 = np.random.choice(indices_2, max_count, replace=True)

        # 4. 合并并打乱
        train_indices = np.concatenate([balanced_idx_0, balanced_idx_1, balanced_idx_2])
        np.random.shuffle(train_indices)
        
    else:
        # 如果不平衡，直接使用自然顺序索引
        train_indices = np.arange(train_size_original)
    
    # 验证集永远不需要平衡，保持原样以便评估真实表现
    val_indices = np.arange(val_size)
    
    # 创建 Dataset - 添加 targets
    train_dataset = ForexClassificationDataset(train_inputs, train_labels, train_targets, train_indices)
    val_dataset = ForexClassificationDataset(val_inputs, val_labels, val_targets, val_indices)
    
    if rank == 0:
        print(f"\nDataset loaded & processed:", flush=True)
        print(f"  Train (Original Samples): {train_size_original:,}", flush=True)
        print(f"  Train (Balanced Indices): {len(train_indices):,} (Virtual Epoch Size)", flush=True)
        print(f"  Val   (Samples):          {val_size:,}", flush=True)
        
        if balance_train:
            print(f"  > Training data is now perfectly balanced (1:1:1).", flush=True)
        
        # 简单的统计显示
        print(f"\nData statistics (Train):", flush=True)
        print(f"  Mean: {train_inputs.mean():.6f}, Std: {train_inputs.std():.6f}", flush=True)
    
    return train_dataset, val_dataset, len(train_indices), val_size


def create_test_dataset(data_path, dtype, rank):
    """
    单独加载测试集
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    raw_data = np.load(data_path)
    
    if 'X_test' not in raw_data or 'y_test' not in raw_data:
        if rank == 0:
            print("Warning: Test set not found in the npz file", flush=True)
        return None, None, 0
    
    test_inputs = torch.from_numpy(raw_data['X_test']).to(dtype)
    test_labels = torch.from_numpy(raw_data['y_test']).long()
    test_targets = torch.from_numpy(raw_data['targets_test']).to(dtype)
    test_size = test_inputs.shape[0]
    
    if rank == 0:
        print(f"\nTest set loaded: {test_size:,} samples", flush=True)
    
    test_indices = np.arange(test_size)
    test_dataset = ForexClassificationDataset(test_inputs, test_labels, test_targets, test_indices)
    
    return test_dataset, test_size


def create_dataloader(dataset, batch_size, num_workers, world_size, rank, shuffle=True):
    # DistributedSampler 会自动处理切分，即使我们的 dataset 索引已经是重采样过的
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    
    per_gpu_workers = max(1, num_workers // world_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=per_gpu_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )
    return loader, sampler