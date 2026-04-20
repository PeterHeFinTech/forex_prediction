
import pandas as pd
from sklearn.model_selection import train_test_split

# ===== Cell 5 =====
import numpy as np
import os
from typing import Tuple, Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_merge_forex_data(ticker_file: str, data_files: Dict[str, str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    加载并合并外汇数据
    
    参数:
        ticker_file: 货币对文件 (如high_pairs.txt)
        data_files: 数据文件字典 {'open': 'open.npz', ...}
    
    返回:
        merged: 合并后的数据 (货币对数, 特征数, 天数)
        pairs: 货币对列表 (如['EURUSD', 'GBPUSD', ...])
        feature_names: 特征名称列表
    """
    print("="*70)
    print("步骤1: 加载并合并外汇数据")
    print("="*70)
    
    # 1. 读取货币对信息
    with open(ticker_file, 'r') as f:
        all_pairs = [line.strip() for line in f.readlines() if line.strip()]

    # 删除 ABC/ABC 形式的无效 pair（如 EUR/EUR, GBP/GBP, USD/USD）
    keep_indices = []
    dropped_pairs = []
    for i, p in enumerate(all_pairs):
        parts = p.split('/')
        if len(parts) == 2 and parts[0] == parts[1]:
            dropped_pairs.append(p)
        else:
            keep_indices.append(i)

    pairs = [all_pairs[i] for i in keep_indices]
    raw_num_pairs = len(all_pairs)
    num_pairs = len(pairs)

    print(f"原始货币对数量: {raw_num_pairs}")
    if dropped_pairs:
        print(f"删除 ABC/ABC 货币对: {dropped_pairs}")
    print(f"清洗后货币对数量: {num_pairs}")
    print(f"前10个货币对: {pairs[:10]}")
    
    # 2. 加载特征数据
    arrays = []
    feature_names = []
    
    for feat_name, file_path in data_files.items():
        print(f"\n加载 {feat_name}: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"  警告: 文件不存在 {file_path}")
            continue
            
        try:
            data = np.load(file_path)
            keys = list(data.keys())
            
            if len(keys) == 0:
                print(f"  错误: 文件为空 {file_path}")
                continue
                
            # 通常只有一个键，取第一个
            key = keys[0]
            array = data[key]
            
            print(f"  原始形状: {array.shape}")
            print(f"  数据类型: {array.dtype}")
            
            # 检查数据形状并自动转置（先对齐到原始 pair 数）
            if array.shape[0] == raw_num_pairs:
                # 形状正确: (货币对数, 天数)
                processed = array
                print(f"  形状正确，无需转置")
            elif array.shape[1] == raw_num_pairs:
                # 需要转置: (天数, 货币对数) -> (货币对数, 天数)
                processed = array.T
                print(f"  检测到需要转置")
            else:
                # 尝试自动检测
                print(f"  尝试自动检测维度...")
                if abs(array.shape[0] - raw_num_pairs) < abs(array.shape[1] - raw_num_pairs):
                    processed = array
                else:
                    processed = array.T
                
                if processed.shape[0] != raw_num_pairs:
                    print(f"  警告: 自动检测后形状仍不匹配: {processed.shape} vs {raw_num_pairs}")

            # 删除 ABC/ABC 对应的数据行
            if processed.shape[0] == raw_num_pairs:
                processed = processed[keep_indices, :]
                print(f"  删除 ABC/ABC 后形状: {processed.shape}")
            
            # 检查NaN值
            nan_count = np.isnan(processed).sum()
            if nan_count > 0:
                print(f"  警告: 发现 {nan_count} 个NaN值 ({nan_count/processed.size:.2%})，将填充为0")

            # 将所有缺失值/无穷值填充为0
            processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 检查零值
            zero_count = np.sum(processed == 0)
            print(f"  零值数量: {zero_count} ({zero_count/processed.size:.2%})")
            
            arrays.append(processed)
            feature_names.append(feat_name)
            
        except Exception as e:
            print(f"  错误: 加载文件 {file_path} 时出错: {str(e)}")
            continue
    
    if len(arrays) == 0:
        raise ValueError("没有成功加载任何数据文件")
    
    # 3. 验证所有数组形状一致
    print(f"\n验证数据形状一致性...")
    base_shape = arrays[0].shape
    
    for i in range(1, len(arrays)):
        if arrays[i].shape != base_shape:
            print(f"  错误: 形状不匹配!")
            print(f"    {feature_names[0]}: {base_shape}")
            print(f"    {feature_names[i]}: {arrays[i].shape}")
            
            # 尝试调整形状
            min_rows = min(base_shape[0], arrays[i].shape[0])
            min_cols = min(base_shape[1], arrays[i].shape[1])
            
            arrays[0] = arrays[0][:min_rows, :min_cols]
            arrays[i] = arrays[i][:min_rows, :min_cols]
            print(f"    调整到相同形状: {min_rows} x {min_cols}")
    
    # 更新货币对列表（如果数据被截断）
    if base_shape[0] != num_pairs:
        print(f"\n注意: 数据形状({base_shape[0]})与货币对数量({num_pairs})不匹配")
        print(f"将截断货币对列表以匹配数据")
        pairs = pairs[:base_shape[0]]
        num_pairs = len(pairs)
        print(f"更新后货币对数量: {num_pairs}")
    
    # 4. 合并数据
    print(f"\n合并数据...")
    merged = np.stack(arrays, axis=1)  # (货币对数, 特征数, 天数)
    
    print(f"\n合并完成:")
    print(f"  数据形状: {merged.shape}")
    print(f"  特征顺序: {feature_names}")
    print(f"  货币对数量: {merged.shape[0]}")
    print(f"  交易天数: {merged.shape[2]}")
    print(f"  总数据点: {merged.size:,}")
    print(f"  内存占用: {merged.nbytes/(1024**3):.2f} GB")
    
    return merged, pairs, feature_names

# 运行步骤1
if __name__ == "__main__":
    # 定义文件路径
    data_files = {
        'open': os.path.join(BASE_DIR, 'open.npz'),
        'high': os.path.join(BASE_DIR, 'high.npz'),
        'low': os.path.join(BASE_DIR, 'low.npz'),
        'close': os.path.join(BASE_DIR, 'close.npz')  # 外汇通常使用close，不是adjusted
    }
    
    merged_data, pairs, features = load_and_merge_forex_data(
        os.path.join(BASE_DIR, 'pairs.txt'),
        data_files
    )
    
    # 保存合并后的数据（包含货币对信息）
    np.savez('forex_merged_data.npz',
             data=merged_data,
             pairs=np.array(pairs, dtype=object),
             features=features,
             data_shape=merged_data.shape)
    
    print(f"\n合并数据已保存到: forex_merged_data.npz")
    
    # 显示详细信息
    print(f"\n数据详细信息:")
    print(f"  货币对示例: {pairs[:5]}...")  # 显示前5个
    print(f"  特征: {features}")
    
    # 检查数据质量
    print(f"\n数据质量检查:")
    for i, pair in enumerate(pairs[:3]):  # 检查前3个货币对
        pair_data = merged_data[i]
        print(f"  {pair}:")
        for j, feat in enumerate(features):
            feat_data = pair_data[j]
            print(f"    {feat}: [{feat_data.min():.6f}, {feat_data.max():.6f}], "
                  f"nan={np.isnan(feat_data).sum()}, zero={(feat_data==0).sum()}")

# ===== Cell 7 =====
merged_data

# ===== Cell 8 =====
import numpy as np
from typing import List, Tuple, Optional

def create_qualified_forex_samples(
        merged: np.ndarray,
        pairs: List[str],
        window_size: int = 128,
        macd_zero_check_len: int = 5,
    min_price_threshold: float = 1e-5,
        sample_time_axis: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    从完整特征数据中切分外汇样本，并在样本中保存货币对信息与时间信息
    
    参数:
        merged: 合并后的数据 (num_pairs, num_features, num_days)
        pairs: 货币对列表
        window_size: 样本窗口长度
        macd_zero_check_len: 检查 MACD 起始零值的长度（>0 时启用）
        sample_time_axis: 时间轴数组，长度应为 num_days（例如 close_timestamps.npy）
        min_price_threshold: 过滤掉最后一个输入时点或目标时点中任一 OHLC 值小于该阈值的样本
        verbose: 是否打印信息
    
    返回:
        samples_array: 样本数组 (num_samples, num_features, window_size)
        pair_indices_array: 样本对应的货币对索引
        pair_names_array: 样本对应的货币对名称
        targets_array: 样本对应的目标值 (OHLC, 下一天)
        start_indices_array: 样本起始索引
        target_indices_array: 样本目标索引
        sample_times_array: 样本目标时点（time）
        sample_pair_info: 样本对应的货币对信息（列表形式）
    """
    
    num_pairs, num_features, num_days = merged.shape

    if sample_time_axis is not None:
        sample_time_axis = np.asarray(sample_time_axis)
        if len(sample_time_axis) != num_days:
            raise ValueError(f"sample_time_axis 长度({len(sample_time_axis)})必须等于 num_days({num_days})")

    qualified_samples = []
    qualified_pair_indices = []
    qualified_pair_names = []
    qualified_targets = []
    qualified_start_indices = []
    qualified_target_indices = []
    qualified_sample_times = []
    sample_pair_info = []

    for pair_idx in range(num_pairs):
        pair_name = pairs[pair_idx]
        pair_data = merged[pair_idx]
        pair_ohlc = pair_data[:4]

        # 0 值只在开头出现：找到首个全 OHLC 非 0 的时间点
        first_valid_day = 0
        while first_valid_day < num_days and np.any(pair_ohlc[:, first_valid_day] == 0):
            first_valid_day += 1

        # 若有效区间不足以形成一个样本，直接跳过该货币对
        if first_valid_day + window_size + 1 >= num_days:
            continue

        for start in range(first_valid_day, num_days - window_size - 1):
            window_start = start
            window_end = start + window_size
            target_day = window_end
            prev_day = target_day - 1

            window_data = pair_data[:, window_start:window_end]
            target_value = pair_data[:4, target_day]
            _ = prev_day

            # 过滤异常低价样本：最后一个输入时点(index=128)或目标时点(index=129)
            # 任一 OHLC 数值小于阈值时，直接跳过该样本
            last_input_value = window_data[:4, -1]
            if np.any(last_input_value < min_price_threshold) or np.any(target_value < min_price_threshold):
                continue

            # MACD 通道为第 5 个通道（索引4），若开头若干步出现 0 则剔除该样本
            if num_features > 4 and macd_zero_check_len > 0:
                head_len = min(macd_zero_check_len, window_size)
                macd_head = window_data[4, :head_len]
                if np.any(macd_head == 0):
                    continue

            # 保存样本
            qualified_samples.append(window_data)
            qualified_pair_indices.append(pair_idx)
            qualified_pair_names.append(pair_name)
            qualified_targets.append(target_value)
            qualified_start_indices.append(start)
            qualified_target_indices.append(target_day)
            if sample_time_axis is None:
                qualified_sample_times.append(target_day)
            else:
                qualified_sample_times.append(sample_time_axis[target_day])
            sample_pair_info.append(pair_name)  # 保存货币对信息

    # 转为 numpy
    samples_array = np.array(qualified_samples, dtype=np.float32)
    pair_indices_array = np.array(qualified_pair_indices)
    pair_names_array = np.array(qualified_pair_names, dtype=object)
    targets_array = np.array(qualified_targets, dtype=np.float32)
    start_indices_array = np.array(qualified_start_indices)
    target_indices_array = np.array(qualified_target_indices)
    sample_times_array = np.array(qualified_sample_times, dtype=object)

    if verbose:
        print(f"样本数量: {samples_array.shape[0]}")
        print(f"样本形状: {samples_array.shape}")
        print(f"样本对应货币对数量: {len(set(sample_pair_info))}")
        print(f"样本时间范围: {sample_times_array[0]} -> {sample_times_array[-1]}")

    return (
        samples_array,
        pair_indices_array,
        pair_names_array,
        targets_array,
        start_indices_array,
        target_indices_array,
        sample_times_array,
        sample_pair_info
    )

# ===== Cell 10 =====
# 读取时间轴（与 close.npz 对齐）
close_timestamps = np.load(os.path.join(BASE_DIR, 'timestamps.npy'), allow_pickle=True)

# ===== Cell 11 =====
import numpy as np

def calculate_ema_numpy(data, span=None, alpha=None):
    """
    NumPy 实现 Pandas 的 ewm(span=span, adjust=False).mean()
    """
    if alpha is None:
        if span is None:
            raise ValueError("Must provide span or alpha")
        alpha = 2 / (span + 1)
    
    n = len(data)
    out = np.zeros(n)
    out[0] = data[0]
    
    # 递归计算: y_t = (1-alpha)*y_{t-1} + alpha*x_t
    # 对于长度 128 的数组，这个循环耗时几乎可以忽略不计
    for i in range(1, n):
        out[i] = (1 - alpha) * out[i-1] + alpha * data[i]
        
    return out

def generate_factors(ohlc_data):
    """
    输入: ohlc_data (128, 4) -> [Open, High, Low, Close]
    输出: factors (128, 6)
    纯 Numpy 实现，去除 Pandas 依赖
    """
    # 提取 Close 列 (第4列，索引为3)
    close = ohlc_data[:, 3]
    length = len(close)
    
    # ================================
    # 1. MACD (12, 26, 9)
    # ================================
    ema12 = calculate_ema_numpy(close, span=12)
    ema26 = calculate_ema_numpy(close, span=26)
    
    macd = ema12 - ema26
    
    # Signal line 是 MACD 的 EMA(9)
    signal = calculate_ema_numpy(macd, span=9)
    
    macd_hist = macd - signal
    
    # ================================
    # 2. RSI (14)
    # ================================
    # 计算一阶差分，并在头部补 0 保持长度对齐
    delta = np.zeros_like(close)
    delta[1:] = close[1:] - close[:-1]
    
    # 获取涨跌幅
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0)) # 取绝对值
    
    # 注意：原 Pandas 代码 RSI 使用 alpha=1/14
    avg_gain = calculate_ema_numpy(gain, alpha=1/14)
    avg_loss = calculate_ema_numpy(loss, alpha=1/14)
    
    # 避免除以 0
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi / 100.0  # 归一化到 0-1
    
    # ================================
    # 3. Bollinger Bands (20)
    # ================================
    window = 20
    
    # 使用 stride_tricks 创建滑动窗口视图
    # 形状会变成 (128-19, 20)
    shape = (length - window + 1, window)
    strides = (close.strides[0], close.strides[0])
    
    # 创建视图 (内存高效，不复制数据)
    windows = np.lib.stride_tricks.as_strided(close, shape=shape, strides=strides)
    
    # 计算均值和标准差
    rolling_mean = np.mean(windows, axis=1)
    rolling_std = np.std(windows, axis=1)
    
    # 计算带宽
    valid_bb_width = (4 * rolling_std) / (rolling_mean + 1e-8)
    
    # 因为滑动窗口会让长度变短，我们需要在前面填充 0
    pad_width = window - 1
    bb_width = np.zeros(length)
    bb_width[pad_width:] = valid_bb_width
    
    # ================================
    # 4. Log Return
    # ================================
    log_ret = np.zeros(length)
    prev_close = close[:-1]
    curr_close = close[1:]

    # 仅在分子分母都 > 0 时计算 log，避免触发 divide by zero warning
    valid = (prev_close > 0) & (curr_close > 0)
    log_vals = np.zeros(length - 1)
    log_vals[valid] = np.log(curr_close[valid] / prev_close[valid])
    log_ret[1:] = log_vals
    
    # ================================
    # 合并输出
    # ================================
    # 堆叠所有因子
    # 顺序: [MACD, Signal, Hist, RSI, BB, LogRet]
    factors = np.column_stack((macd, signal, macd_hist, rsi, bb_width, log_ret))
    
    # 填充可能的 NaN / Inf 为 0
    factors = np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)
    
    return factors.astype(np.float32, copy=False)

# ===== Cell 12 =====
import numpy as np

def create_full_features_before_sampling(merged_ohlc: np.ndarray) -> np.ndarray:
    """
    先在全量时间轴上计算因子，再与 OHLC 拼接。
    输入: merged_ohlc (num_pairs, 4, num_days)
    输出: merged_with_factors (num_pairs, 10, num_days)
    """
    ohlc = merged_ohlc[:, :4, :].astype(np.float32, copy=False)
    num_pairs, _, num_days = ohlc.shape

    factors_full = np.zeros((num_pairs, 6, num_days), dtype=np.float32)

    for pair_idx in range(num_pairs):
        ohlc_data = ohlc[pair_idx].T  # (num_days, 4)
        factors = generate_factors(ohlc_data)  # (num_days, 6)
        factors_full[pair_idx] = factors.T

    merged_with_factors = np.concatenate([ohlc, factors_full], axis=1).astype(np.float32, copy=False)
    return merged_with_factors


full_feature_data = create_full_features_before_sampling(merged_data)

(samples,
 pair_indices,
 pair_names,
 targets,
 start_indices,
 target_indices,
 sample_times,
 sample_pair_info) = create_qualified_forex_samples(
    merged=full_feature_data,
    pairs=pairs,
    window_size=128,
    macd_zero_check_len=5,
    sample_time_axis=close_timestamps,
    verbose=True
)

print(f"\n处理完成！")
print(f"全量特征形状: {full_feature_data.shape}")
print(f"切窗后样本形状: {samples.shape}")

# ===== Cell 14 =====
import numpy as np
import os
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split

# ==================== 工具函数 ====================
def create_atr_labels(samples: np.ndarray, 
                     targets: np.ndarray,
                     atr_multiplier: float = 0.5,
                     atr_period: int = 128,
                     verbose: bool = True) -> np.ndarray:
    """
    基于ATR生成标签 (0:大跌, 1:平盘, 2:大涨)
    只返回标签，不保留具体价格
    """
    num_samples = samples.shape[0]
    labels = np.zeros(num_samples, dtype=np.int32)
    
    # 特征索引 - OHLC
    HIGH_IDX = 1
    LOW_IDX = 2
    CLOSE_IDX = 3
    
    # 统计
    label_counts = {0: 0, 1: 0, 2: 0}
    
    for i in range(num_samples):
        sample = samples[i]  # (特征数, window_size)
        
        # 提取数据
        high_series = sample[HIGH_IDX]
        low_series = sample[LOW_IDX]
        close_series = sample[CLOSE_IDX]
        
        # 计算收益率
        last_close = close_series[-1]
        target_close = targets[i, CLOSE_IDX]
        
        if last_close > 0:
            return_rate = (target_close - last_close) / last_close
        else:
            return_rate = 0.0
        
        # 计算ATR
        n = len(close_series)
        tr = np.zeros(n)
        tr[0] = high_series[0] - low_series[0]
        
        for j in range(1, n):
            hl = high_series[j] - low_series[j]
            hc = abs(high_series[j] - close_series[j-1])
            lc = abs(low_series[j] - close_series[j-1])
            tr[j] = max(hl, hc, lc)
        
        # 计算ATR
        if atr_period <= n:
            atr = np.mean(tr[-atr_period:])
        else:
            atr = np.mean(tr)
        
        # 计算阈值
        if last_close > 0:
            threshold = (atr_multiplier * atr) / last_close
        else:
            threshold = 0
        
        # 打标签
        if return_rate > threshold:
            labels[i] = 2  # 大涨
            label_counts[2] += 1
        elif return_rate < -threshold:
            labels[i] = 0  # 大跌
            label_counts[0] += 1
        else:
            labels[i] = 1  # 平盘
            label_counts[1] += 1
    
    if verbose:
        print(f"ATR标签生成:")
        print(f"  样本数: {num_samples:,}")
        print(f"  标签分布: 大跌({label_counts[0]}) 平盘({label_counts[1]}) 大涨({label_counts[2]})")
        print(f"  比例: 大跌({label_counts[0]/num_samples:.1%}) 平盘({label_counts[1]/num_samples:.1%}) 大涨({label_counts[2]/num_samples:.1%})")
    
    return labels

def create_std_labels(samples: np.ndarray,
                     targets: np.ndarray,
                     std_multiplier: float = 0.5,
                     std_period: int = 128,
                     verbose: bool = True) -> np.ndarray:
    """
    基于标准差生成标签 (0:大跌, 1:平盘, 2:大涨)
    """
    num_samples = samples.shape[0]
    labels = np.zeros(num_samples, dtype=np.int32)
    
    # 特征索引
    CLOSE_IDX = 3
    
    # 统计
    label_counts = {0: 0, 1: 0, 2: 0}
    
    for i in range(num_samples):
        sample = samples[i]
        close_series = sample[CLOSE_IDX]
        
        # 计算收益率
        last_close = close_series[-1]
        target_close = targets[i, CLOSE_IDX]
        
        if last_close > 0:
            return_rate = (target_close - last_close) / last_close
        else:
            return_rate = 0.0
        
        # 计算历史收益率的标准差
        if len(close_series) >= std_period + 1:
            hist_returns = (close_series[1:] - close_series[:-1]) / close_series[:-1]
            
            if len(hist_returns) >= std_period:
                recent_returns = hist_returns[-std_period:]
            else:
                recent_returns = hist_returns
            
            if len(recent_returns) > 0:
                threshold = std_multiplier * np.std(recent_returns)
            else:
                threshold = 0
        else:
            threshold = 0
        
        # 打标签
        if return_rate > threshold:
            labels[i] = 2
            label_counts[2] += 1
        elif return_rate < -threshold:
            labels[i] = 0
            label_counts[0] += 1
        else:
            labels[i] = 1
            label_counts[1] += 1
    
    if verbose:
        print(f"标准差标签生成:")
        print(f"  样本数: {num_samples:,}")
        print(f"  标签分布: 大跌({label_counts[0]}) 平盘({label_counts[1]}) 大涨({label_counts[2]})")
        print(f"  比例: 大跌({label_counts[0]/num_samples:.1%}) 平盘({label_counts[1]/num_samples:.1%}) 大涨({label_counts[2]/num_samples:.1%})")
    
    return labels

# ===== Cell 15 =====
# 假设你已经有：
# samples, targets, pair_indices, pair_names, start_indices, target_indices

print("生成 ATR 标签...")
atr_labels = create_atr_labels(samples, targets, atr_multiplier=0.5, atr_period=128, verbose=True)

# print("\n生成标准差标签...")
# std_labels = create_std_labels(samples, targets, std_multiplier=0.5, std_period=128, verbose=True)

# ===== Cell 16 =====
import os
import numpy as np

def create_dataset(X: np.ndarray, y: np.ndarray, targets: np.ndarray,
                  train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray,
                  output_file: str, dataset_name: str, pair_ids: np.ndarray, pair_names_per_sample: np.ndarray,
                  sample_times: np.ndarray, factor_keys: list,
                  keep_last_len: Optional[int] = None,
                  compress: bool = False):
    """
    创建并保存一个完整的数据集
    targets: 第129天的OHLC数据 (num_samples, 4)
    额外保存: 每个样本的 pair 名称与时间戳
    """
    print(f"\n[{dataset_name}] 正在创建数据集...")
    
    # 应用 Mask，按需保留最后 keep_last_len 个时间步（None 表示不截断）
    def _select_x(mask: np.ndarray) -> np.ndarray:
        x_part = X[mask]
        if keep_last_len is not None:
            x_part = x_part[:, -keep_last_len:, :]
        return x_part.astype(np.float32, copy=False)

    X_train = _select_x(train_mask)
    X_val = _select_x(val_mask)
    X_test = _select_x(test_mask)

    y_train = y[train_mask].astype(np.int32)
    targets_train = targets[train_mask].astype(np.float32)  # 第129天价格

    y_val   = y[val_mask].astype(np.int32)
    targets_val = targets[val_mask].astype(np.float32)

    y_test  = y[test_mask].astype(np.int32)
    targets_test = targets[test_mask].astype(np.float32)
    
    # 记录每个集合对应的 Pair 与 Time
    pairs_train = pair_ids[train_mask]
    pairs_val   = pair_ids[val_mask]
    pairs_test  = pair_ids[test_mask]

    pair_names_train = pair_names_per_sample[train_mask].astype(object)
    pair_names_val   = pair_names_per_sample[val_mask].astype(object)
    pair_names_test  = pair_names_per_sample[test_mask].astype(object)

    timestamp_train = sample_times[train_mask].astype(object)
    timestamp_val   = sample_times[val_mask].astype(object)
    timestamp_test  = sample_times[test_mask].astype(object)
    
    print(f"  训练集: {X_train.shape[0]:,} 样本")
    print(f"  验证集: {X_val.shape[0]:,} 样本")
    print(f"  测试集: {X_test.shape[0]:,} 样本")
    if keep_last_len is None:
        print(f"  序列长度: {X_train.shape[1]} (保留完整长度)")
    else:
        print(f"  序列长度: {X_train.shape[1]} (只保留最后 {keep_last_len})")
    
    # 打印标签分布
    def print_label_distribution(labels, name):
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  {name}标签分布:", end="")
        for label, count in zip(unique, counts):
            # 假设 0:大跌, 1:平盘, 2:大涨
            label_map = {0: '大跌', 1: '平盘', 2: '大涨'}
            label_name = label_map.get(label, str(label))
            print(f" {label_name}({label})={count:,}", end="")
        print()
    
    print_label_distribution(y_train, "训练集")
    print_label_distribution(y_val, "验证集")
    print_label_distribution(y_test, "测试集")
    
    feature_names = ['open', 'high', 'low', 'close'] + factor_keys
    target_feature_names = ['target_open', 'target_high', 'target_low', 'target_close']
    
    # 保存 - 添加 targets / pair_name / timestamp 数据
    # 注：压缩写盘会显著增加峰值内存，内存紧张时建议 compress=False
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        output_file,
        X_train=X_train, y_train=y_train, targets_train=targets_train,
        X_val=X_val,     y_val=y_val,     targets_val=targets_val,
        X_test=X_test,   y_test=y_test,   targets_test=targets_test,
        pairs_train=pairs_train,
        pairs_val=pairs_val,
        pairs_test=pairs_test,
        pair_names_train=pair_names_train,
        pair_names_val=pair_names_val,
        pair_names_test=pair_names_test,
        timestamp_train=timestamp_train,
        timestamp_val=timestamp_val,
        timestamp_test=timestamp_test,
        feature_names=feature_names,
        target_feature_names=target_feature_names,
        label_names=['大跌', '平盘', '大涨'],
        dataset_name=dataset_name,
        generation_timestamp=str(np.datetime64('now'))
    )
    
    print(f"  已保存到: {output_file}")
    file_size = os.path.getsize(output_file) / (1024**3)
    print(f"  文件大小: {file_size:.2f} GB")
    print(f"  targets形状: train={targets_train.shape}, val={targets_val.shape}, test={targets_test.shape}")
    print(f"  pair_name示例: train[0]={pair_names_train[0] if len(pair_names_train) > 0 else 'N/A'}")
    print(f"  time示例: train[0]={timestamp_train[0] if len(timestamp_train) > 0 else 'N/A'}")
    
    return output_file


def verify_datasets():
    datasets = ['forex_atr_by_pair.npz', 'forex_atr_by_time.npz',
                'forex_std_by_pair.npz', 'forex_std_by_time.npz']
    
    print("\n" + "="*80)
    print("数据集验证")
    print("="*80)
    
    for dataset_file in datasets:
        if os.path.exists(dataset_file):
            print(f"\n验证数据集: {dataset_file}")
            data = np.load(dataset_file, allow_pickle=True)
            
            print(f"  特征形状: X_train={data['X_train'].shape}")
            print(f"  标签形状: y_train={data['y_train'].shape}")
            if 'targets_train' in data:
                print(f"  目标价格形状: targets_train={data['targets_train'].shape}")
            if 'pair_names_train' in data:
                print(f"  pair名称示例: {data['pair_names_train'][:3]}")
            if 'timestamp_train' in data:
                print(f"  time示例: {data['timestamp_train'][:3]}")
            print(f"  数据类型: X={data['X_train'].dtype}, y={data['y_train'].dtype}")
            unique_labels = np.unique(data['y_train'])
            print(f"  标签范围: {unique_labels.tolist()}")
            
            total_mb = (data['X_train'].nbytes + data['X_val'].nbytes + 
                       data['X_test'].nbytes) / (1024**2)
            print(f"  内存占用: {total_mb:.1f} MB")
        else:
            print(f"\n警告: 文件不存在 {dataset_file}")

# ===== Cell 18 =====
from sklearn.model_selection import train_test_split

n_samples = len(samples)

# ==========================================
# 策略 A: By Pair (按货币对切分) - 7:1:2
# ==========================================
# 假设 sample_pairs 包含了每个样本对应的 pair_idx
# unique_pairs = np.unique(sample_pairs) 
unique_pairs = np.unique(pair_indices) # 使用你代码里的变量名

# 1. 划分货币对 ID
# 先分出 70% 训练集，剩余 30% 临时
train_pairs, temp_pairs = train_test_split(unique_pairs, test_size=0.3, random_state=42)
# 再把 30% 分为 10% Val 和 20% Test (0.3 * 0.333 = 0.1, 0.3 * 0.667 = 0.2)
val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.6667, random_state=42)

# 2. 生成 Mask
train_mask_pair = np.isin(pair_indices, train_pairs)
val_mask_pair   = np.isin(pair_indices, val_pairs)
test_mask_pair  = np.isin(pair_indices, test_pairs)

print(f"By Pair Mask 统计: Train={train_mask_pair.sum()}, Val={val_mask_pair.sum()}, Test={test_mask_pair.sum()}")


# ==========================================
# 策略 B: By Time (按时间排序切分) - 7:1:2
# ==========================================
# 必须先按时间对样本进行排序，才能保证 Train < Val < Test
# 假设 sample_times 包含了每个样本的 start_time (int 或 datetime)
# 如果你没有 sample_times，你需要从生成循环中收集它

# 1. 获取按时间排序后的索引
sorted_indices = np.argsort(start_indices)  # 使用你代码里的变量名

# 2. 计算切分点数量
n_train = int(n_samples * 0.6)
n_val   = int(n_samples * 0.2) # 10%
# 剩余 20% 为 Test

# 3. 获取对应的索引 ID
train_indices_sorted = sorted_indices[:n_train]
val_indices_sorted   = sorted_indices[n_train : n_train + n_val]
test_indices_sorted  = sorted_indices[n_train + n_val :]

# 4. 初始化 Mask
train_mask_time = np.zeros(n_samples, dtype=bool)
val_mask_time   = np.zeros(n_samples, dtype=bool)
test_mask_time  = np.zeros(n_samples, dtype=bool)

# 5. 填充 Mask
train_mask_time[train_indices_sorted] = True
val_mask_time[val_indices_sorted]     = True
test_mask_time[test_indices_sorted]   = True

# 6. 【关键】添加 Embargo (隔离带/净化)
# 防止 Train 的末尾窗口与 Val 的开头窗口重叠
window_size = 128

# 去掉 Val 集合中时间最早的 window_size 个样本 (它们与 Train 尾部重叠)
if len(val_indices_sorted) > window_size:
    purge_val_indices = val_indices_sorted[:window_size]
    val_mask_time[purge_val_indices] = False

# 去掉 Test 集合中时间最早的 window_size 个样本 (它们与 Val 尾部重叠)
if len(test_indices_sorted) > window_size:
    purge_test_indices = test_indices_sorted[:window_size]
    test_mask_time[purge_test_indices] = False

print(f"By Time Mask 统计: Train={train_mask_time.sum()}, Val={val_mask_time.sum()}, Test={test_mask_time.sum()}")

# ===== Cell 20 =====
# 统一转置 Input (N, L, C) -> (N, C, L) 
# 注意：PyTorch CNN通常喜欢 (N, C, L)，如果是 LSTM/Transformer 通常保持 (N, L, C)
# 这里沿用你原本的 transpose(0,2,1)
X_data = samples.transpose(0, 2, 1)

# # -------------------------------------------------------
# # 1. 保存 ATR by Pair (Train: Pair A-M, Test: Pair N-Z)
# # -------------------------------------------------------
# create_dataset(X_data, atr_labels, targets,
#                train_mask_pair, val_mask_pair, test_mask_pair,
#                'forex_atr_by_pair.npz', 'atr_by_pair', 
#                pair_ids=pair_indices, pair_names_per_sample=pair_names, sample_times=sample_times, factor_keys=[], keep_last_len=96)

# -------------------------------------------------------
# 2. 保存 ATR by Time (Train: 2010-2018, Test: 2020-2023)
# -------------------------------------------------------
create_dataset(X_data, atr_labels, targets,
               train_mask_time, val_mask_time, test_mask_time,
               'forex_atr_by_time.npz', 'atr_by_time', 
               pair_ids=pair_indices, pair_names_per_sample=pair_names, sample_times=sample_times,
               factor_keys=[], keep_last_len=96, compress=False)

# # -------------------------------------------------------
# # 3. 保存 STD by Pair
# # -------------------------------------------------------
# create_dataset(X_data, std_labels, targets,
#                train_mask_pair, val_mask_pair, test_mask_pair,
#                'forex_std_by_pair.npz', 'std_by_pair', 
#                pair_ids=pair_indices, pair_names_per_sample=pair_names, sample_times=sample_times, factor_keys=[], keep_last_len=96)

# # -------------------------------------------------------
# # 4. 保存 STD by Time
# # -------------------------------------------------------
# create_dataset(X_data, std_labels, targets,
#                train_mask_time, val_mask_time, test_mask_time,
#                'forex_std_by_time.npz', 'std_by_time', 
#                pair_ids=pair_indices, pair_names_per_sample=pair_names, sample_times=sample_times, factor_keys=[], keep_last_len=96)

# ===== Cell 21 =====
verify_datasets()

# ===== Cell 22 =====
import numpy as np
import os

def inspect_and_verify(file_list):
    for file_name in file_list:
        if not os.path.exists(file_name):
            print(f"❌ 文件不存在: {file_name}")
            continue
            
        print(f"\n{'='*30}")
        print(f"📂 正在检查: {file_name}")
        print(f"{'='*30}")
        
        data = np.load(file_name, allow_pickle=True)
        
        # 1. 基础信息
        X_train = data['X_train']
        y_train = data['y_train']
        feature_names = data['feature_names']
        
        print(f"📊 训练集形状: {X_train.shape}")
        print(f"🏷️  特征列表: {feature_names}")
        
        # 尝试获取 Pair ID 来验证划分逻辑
        try:
            pairs_train = data['pairs_train']
            pairs_test = data['pairs_test']
            
            # --- 验证划分逻辑 ---
            train_unique = set(np.unique(pairs_train))
            test_unique = set(np.unique(pairs_test))
            intersection = train_unique.intersection(test_unique)
            
            if "by_pair" in file_name:
                if len(intersection) == 0:
                    print(f"✅ [验证通过] By Pair: 训练集和测试集的货币对完全隔离 (0 重叠)")
                else:
                    print(f"❌ [可能有误] By Pair: 发现 {len(intersection)} 个重叠货币对!")
            elif "by_time" in file_name:
                if len(intersection) > 0:
                    print(f"✅ [验证通过] By Time: 训练集和测试集包含相同的货币对 (属于正常，因为是按时间切的)")
                else:
                    print(f"⚠️ [注意] By Time: 训练集和测试集货币对完全不同 (这很少见，除非数据量极小)")
        except KeyError:
            print("⚠️ 未找到 pairs_train 数据，无法验证划分逻辑 (可能是旧版代码生成的)")

        # 2. 打印前 5 个样本的内容
        print(f"\n🔎 --- 前 5 个样本预览 ---")
        for i in range(5):
            print(f"\n🔹 样本 #{i}")
            
            
            # 打印标签
            label_map = {0: '大跌', 1: '平盘', 2: '大涨'}
            label = y_train[i]
            print(f"   标签: {label} ({label_map.get(label, '未知')})")
            
            # 打印 OHLC 数据 (取前 3 个时间步)
            # 假设数据形状是 (Channels, Length) -> (4, window_size)
            sample_data = X_train[i]
            
            # 简单的转置处理，确保打印出来是 (Time, Feature)
            if sample_data.shape[0] < sample_data.shape[1]: 
                sample_data = sample_data.T
            
            print(f"   数据 (前3行):")
            # 打印列头
            print(f"   {'Open':>10} {'High':>10} {'Low':>10} {'Close':>10}")
            for row in sample_data[:3]:
                # 假设前4列是 OHLC
                print(f"   {row[0]:10.5f} {row[1]:10.5f} {row[2]:10.5f} {row[3]:10.5f} {row[4:]}")

# 定义要检查的文件
files = [
    'forex_atr_by_pair.npz', 
    'forex_atr_by_time.npz',
    'forex_std_by_pair.npz', 
    'forex_std_by_time.npz'
]

inspect_and_verify(files)
