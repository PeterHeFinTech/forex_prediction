#!/usr/bin/env python3
"""
验证数据集是否包含所需的 targets 字段
"""
import numpy as np
import sys

def verify_dataset(data_path):
    """验证数据集结构"""
    print(f"\n验证数据集: {data_path}")
    print("="*80)
    
    try:
        data = np.load(data_path)
        keys = list(data.keys())
        print(f"\n找到的keys: {keys}")
        
        # 检查必需的keys
        required_keys = ['X_train', 'y_train', 'targets_train',
                        'X_val', 'y_val', 'targets_val',
                        'X_test', 'y_test', 'targets_test']
        
        missing_keys = [k for k in required_keys if k not in keys]
        if missing_keys:
            print(f"\n❌ 缺少以下keys: {missing_keys}")
            return False
        
        print(f"\n✅ 所有必需的keys都存在")
        
        # 检查数据形状
        print(f"\n数据形状:")
        for split in ['train', 'val', 'test']:
            X_key = f'X_{split}'
            y_key = f'y_{split}'
            targets_key = f'targets_{split}'
            
            X_shape = data[X_key].shape
            y_shape = data[y_key].shape
            targets_shape = data[targets_key].shape
            
            print(f"\n{split.upper()}:")
            print(f"  {X_key}: {X_shape}")
            print(f"  {y_key}: {y_shape}")
            print(f"  {targets_key}: {targets_shape}")
            
            # 验证形状一致性
            N = X_shape[0]
            if y_shape[0] != N:
                print(f"  ❌ 标签数量 ({y_shape[0]}) 与样本数量 ({N}) 不匹配")
                return False
            if targets_shape[0] != N:
                print(f"  ❌ targets数量 ({targets_shape[0]}) 与样本数量 ({N}) 不匹配")
                return False
            if len(targets_shape) != 2 or targets_shape[1] != 4:
                print(f"  ❌ targets形状应该是 (N, 4)，但得到 {targets_shape}")
                return False
            if len(X_shape) != 3 or X_shape[1] != 128 or X_shape[2] != 10:
                print(f"  ❌ X形状应该是 (N, 128, 10)，但得到 {X_shape}")
                return False
            
            print(f"  ✅ 形状正确")
        
        # 验证收益计算逻辑
        print(f"\n验证收益计算:")
        X_test = data['X_test']
        targets_test = data['targets_test']
        
        # 随机选择一个样本
        idx = np.random.randint(0, len(X_test))
        
        # 获取第128天和第129天的close价格
        price_128 = X_test[idx, -1, 3]  # 最后一个时间步的close (索引3)
        price_129 = targets_test[idx, 3]  # 目标OHLC的close (索引3)
        
        # 计算收益
        actual_return = (price_129 - price_128) / price_128 * 100
        
        print(f"  样本索引: {idx}")
        print(f"  第128天收盘价: {price_128:.5f}")
        print(f"  第129天收盘价: {price_129:.5f}")
        print(f"  实际收益率: {actual_return:+.4f}%")
        
        # 检查价格是否合理
        if price_128 <= 0 or price_129 <= 0:
            print(f"  ❌ 价格不应该为负数或零")
            return False
        
        if abs(actual_return) > 50:
            print(f"  ⚠️  收益率异常大 (>{actual_return:.2f}%)，请检查数据")
        else:
            print(f"  ✅ 收益率在合理范围内")
        
        print(f"\n" + "="*80)
        print(f"✅ 数据集验证通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python verify_dataset.py <数据集路径>")
        print("\n示例:")
        print("  python verify_dataset.py forex/fin_factor/forex_atr_by_time.npz")
        sys.exit(1)
    
    data_path = sys.argv[1]
    success = verify_dataset(data_path)
    sys.exit(0 if success else 1)
