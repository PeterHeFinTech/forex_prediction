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
import zipfile
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
    def __init__(self, inputs, labels, targets, time_ids, indices):
        self.x = inputs
        self.y = labels
        self.targets = targets  # 第129天的OHLC价格
        self.time_ids = time_ids  # 每个样本所属时间桶（用于按天聚合评估）
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 通过重采样后的索引列表来访问原始数据
        # 这样即使数据被多次采样，也不需要复制物理内存中的 Tensor
        i = self.indices[idx]
        return self.x[i], self.y[i], self.targets[i], self.time_ids[i]

def sample_tensor_quadruplet(inputs, labels, targets, time_ids, fraction):
    total_size = inputs.shape[0]
    sample_size = max(1, int(total_size * fraction))
    sample_indices = np.random.choice(total_size, size=sample_size, replace=False)
    return inputs[sample_indices], labels[sample_indices], targets[sample_indices], time_ids[sample_indices], sample_size


def _to_time_ids(arr):
    """
    将任意 timestamp 数组映射为 int64 time_id，便于 DataLoader 和按天聚合。
    """
    arr = np.asarray(arr).reshape(-1)

    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype('datetime64[s]').astype(np.int64)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.int64)

    # 字符串/对象类型：按唯一值编码，保持同一时间戳映射到同一 id
    _, inverse = np.unique(arr.astype(str), return_inverse=True)
    return inverse.astype(np.int64)


def _build_split_time_ids(raw_data, train_size, val_size, test_size, rank):
    """
    构建 train/val/test 对齐的 time_id。优先使用 split 级字段；
    若只有 timestamp，则按可推断规则切分；否则退化为样本索引。
    """
    split_key_candidates = [
        ('timestamp_train', 'timestamp_val', 'timestamp_test'),
        ('timestamps_train', 'timestamps_val', 'timestamps_test'),
        ('time_train', 'time_val', 'time_test'),
        ('times_train', 'times_val', 'times_test'),
    ]

    for k_train, k_val, k_test in split_key_candidates:
        if all(k in raw_data for k in (k_train, k_val, k_test)):
            return (
                _to_time_ids(raw_data[k_train]),
                _to_time_ids(raw_data[k_val]),
                _to_time_ids(raw_data[k_test]),
            )

    if 'timestamp' in raw_data:
        ts = np.asarray(raw_data['timestamp']).reshape(-1)
        total = train_size + val_size + test_size

        if len(ts) == total:
            ts_train = ts[:train_size]
            ts_val = ts[train_size:train_size + val_size]
            ts_test = ts[train_size + val_size:]
            return _to_time_ids(ts_train), _to_time_ids(ts_val), _to_time_ids(ts_test)

        if rank == 0:
            print(
                f"[Timestamp Warning] Found 'timestamp' with length={len(ts)}, "
                f"cannot align to splits ({train_size}, {val_size}, {test_size}). "
                f"Timestamp appears to be dataset-level metadata (e.g., generation time), "
                f"not per-sample time. Daily aggregation will be disabled.",
                flush=True,
            )

    # fallback：使用哨兵 time_id（全部为 -1），显式禁用按天聚合，避免伪时间结果
    return (
        np.full(train_size, -1, dtype=np.int64),
        np.full(val_size, -1, dtype=np.int64),
        np.full(test_size, -1, dtype=np.int64),
    )

def create_dataset(data_path, dtype, rank, balance_train=True, data_fraction=1.0):
    """
    参数:
    balance_train (bool): 是否对训练集进行类别平衡（过采样）。默认为 True。
    data_fraction (float): 要使用的训练数据的比例，范围 0.0-1.0。默认为 1.0（使用全部）。
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    if not zipfile.is_zipfile(data_path):
        raise ValueError(
            f"Dataset file is corrupted or incomplete (not a valid NPZ/ZIP archive): {data_path}"
        )
    
    # 验证data_fraction的有效性
    if not (0.0 < data_fraction <= 1.0):
        raise ValueError(f"data_fraction must be in (0.0, 1.0], got {data_fraction}")
    
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

    test_size_for_time = raw_data['X_test'].shape[0] if 'X_test' in raw_data else 0
    train_time_ids_np, val_time_ids_np, _ = _build_split_time_ids(
        raw_data,
        train_inputs.shape[0],
        val_inputs.shape[0],
        test_size_for_time,
        rank,
    )

    train_time_ids = torch.from_numpy(train_time_ids_np).long()
    val_time_ids = torch.from_numpy(val_time_ids_np).long()
    
    train_size_original = train_inputs.shape[0]
    val_size_original = val_inputs.shape[0]

    # --- 数据子采样：根据data_fraction随机选择训练/验证数据 ---
    if data_fraction < 1.0:
        train_inputs, train_labels, train_targets, train_time_ids, train_size_original = sample_tensor_quadruplet(
            train_inputs, train_labels, train_targets, train_time_ids, data_fraction
        )
        val_inputs, val_labels, val_targets, val_time_ids, val_size = sample_tensor_quadruplet(
            val_inputs, val_labels, val_targets, val_time_ids, data_fraction
        )
        
        if rank == 0:
            print(
                f"\n[Data Sampling] Using {data_fraction*100:.2f}% of training data "
                f"({train_size_original:,}/{raw_data['X_train'].shape[0]:,} samples)",
                flush=True,
            )
            print(
                f"[Data Sampling] Using {data_fraction*100:.2f}% of validation data "
                f"({val_size:,}/{val_size_original:,} samples)",
                flush=True,
            )
    else:
        val_size = val_size_original

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
    train_dataset = ForexClassificationDataset(train_inputs, train_labels, train_targets, train_time_ids, train_indices)
    val_dataset = ForexClassificationDataset(val_inputs, val_labels, val_targets, val_time_ids, val_indices)
    
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


def create_test_dataset(data_path, dtype, rank, data_fraction=1.0):
    """
    单独加载测试集
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    if not zipfile.is_zipfile(data_path):
        raise ValueError(
            f"Dataset file is corrupted or incomplete (not a valid NPZ/ZIP archive): {data_path}"
        )
    if not (0.0 < data_fraction <= 1.0):
        raise ValueError(f"data_fraction must be in (0.0, 1.0], got {data_fraction}")
    
    raw_data = np.load(data_path)
    
    if 'X_test' not in raw_data or 'y_test' not in raw_data:
        if rank == 0:
            print("Warning: Test set not found in the npz file", flush=True)
        return None, None, 0
    
    test_inputs = torch.from_numpy(raw_data['X_test']).to(dtype)
    test_labels = torch.from_numpy(raw_data['y_test']).long()
    test_targets = torch.from_numpy(raw_data['targets_test']).to(dtype)

    train_size_for_time = raw_data['X_train'].shape[0] if 'X_train' in raw_data else 0
    val_size_for_time = raw_data['X_val'].shape[0] if 'X_val' in raw_data else 0
    _, _, test_time_ids_np = _build_split_time_ids(
        raw_data,
        train_size_for_time,
        val_size_for_time,
        test_inputs.shape[0],
        rank,
    )
    test_time_ids = torch.from_numpy(test_time_ids_np).long()
    test_size_original = test_inputs.shape[0]

    if data_fraction < 1.0:
        test_inputs, test_labels, test_targets, test_time_ids, test_size = sample_tensor_quadruplet(
            test_inputs, test_labels, test_targets, test_time_ids, data_fraction
        )
        if rank == 0:
            print(
                f"[Data Sampling] Using {data_fraction*100:.2f}% of test data "
                f"({test_size:,}/{test_size_original:,} samples)",
                flush=True,
            )
    else:
        test_size = test_size_original
    
    if rank == 0:
        print(f"\nTest set loaded: {test_size:,} samples", flush=True)
    
    test_indices = np.arange(test_size)
    test_dataset = ForexClassificationDataset(test_inputs, test_labels, test_targets, test_time_ids, test_indices)
    
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