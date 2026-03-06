#!/usr/bin/env python3
"""
Script to inspect the forex dataset structure and label assignments.
Usage: python3 inspect_data.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Load the data
print("Loading data from: ./fin_factor/forex_atr_by_time.npz")
data = np.load('./fin_factor/forex_atr_by_time.npz')

print("\n" + "="*80)
print("DATASET STRUCTURE")
print("="*80)
print(f"Available keys: {data.files}")
print(f"\nDataset name: {data['dataset_name']}")
print(f"Feature names: {data['feature_names']}")
print(f"Label names: {data['label_names']}")

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

print(f"\nData shapes:")
print(f"  X_train: {X_train.shape} (samples, timesteps, features)")
print(f"  y_train: {y_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  y_val: {y_val.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

print("\n" + "="*80)
print("ANALYZING PRICE DATA AND LABELS")
print("="*80)

def analyze_split(X, y, split_name):
    """Analyze a data split (train/val/test)"""
    print(f"\n{split_name} Set Analysis:")
    print("-" * 60)
    
    # Extract prices
    prices_127 = X[:, 126, 3]  # Day 127 close (index 126)
    prices_128 = X[:, 127, 3]  # Day 128 close (index 127)
    returns = (prices_128 - prices_127) / prices_127 * 100
    
    print(f"Total samples: {len(X):,}")
    
    print(f"\nActual Returns Statistics:")
    print(f"  Mean:   {np.mean(returns):>8.4f}%")
    print(f"  Median: {np.median(returns):>8.4f}%")
    print(f"  Std:    {np.std(returns):>8.4f}%")
    print(f"  Min:    {np.min(returns):>8.4f}%")
    print(f"  Max:    {np.max(returns):>8.4f}%")
    
    print(f"\nLabel Distribution vs Actual Returns:")
    print(f"{'Label':<10} {'Name':<10} {'Count':>10} {'Percentage':>12} {'Avg Return':>12}")
    print("-" * 60)
    
    for i, label_name in enumerate(data['label_names']):
        mask = (y == i)
        count = np.sum(mask)
        pct = count / len(y) * 100
        avg_return = np.mean(returns[mask]) if count > 0 else 0
        print(f"{i:<10} {label_name:<10} {count:>10,} {pct:>11.2f}% {avg_return:>11.4f}%")
    
    # Show return ranges for each label
    print(f"\nReturn Range by Label:")
    for i, label_name in enumerate(data['label_names']):
        mask = (y == i)
        if np.sum(mask) > 0:
            label_returns = returns[mask]
            print(f"  {label_name}: [{np.min(label_returns):.4f}%, {np.max(label_returns):.4f}%]")
    
    return returns, y

# Analyze each split
train_returns, train_labels = analyze_split(X_train, y_train, "TRAIN")
val_returns, val_labels = analyze_split(X_val, y_val, "VALIDATION")
test_returns, test_labels = analyze_split(X_test, y_test, "TEST")

print("\n" + "="*80)
print("SAMPLE EXAMPLES")
print("="*80)

print("\nRandom 20 samples from training set:")
print(f"{'Idx':<8} {'Price_127':<12} {'Price_128':<12} {'Return':<10} {'Label':<8} {'Label_Name'}")
print("-" * 70)
np.random.seed(42)
sample_indices = np.random.choice(len(X_train), 20, replace=False)
for idx in sample_indices:
    price_127 = X_train[idx, 126, 3]
    price_128 = X_train[idx, 127, 3]
    ret = (price_128 - price_127) / price_127 * 100
    label = y_train[idx]
    label_name = data['label_names'][label]
    print(f"{idx:<8} {price_127:<12.5f} {price_128:<12.5f} {ret:>+9.4f}% {label:<8} {label_name}")

print("\n" + "="*80)
print("CHECKING LABEL CONSISTENCY")
print("="*80)

# Find extreme cases
print("\nTop 10 LARGEST POSITIVE returns:")
print(f"{'Return':<12} {'Label':<8} {'Label_Name'}")
print("-" * 40)
top_positive_idx = np.argsort(train_returns)[-10:][::-1]
for idx in top_positive_idx:
    print(f"{train_returns[idx]:>+10.4f}% {train_labels[idx]:<8} {data['label_names'][train_labels[idx]]}")

print("\nTop 10 LARGEST NEGATIVE returns:")
print(f"{'Return':<12} {'Label':<8} {'Label_Name'}")
print("-" * 40)
top_negative_idx = np.argsort(train_returns)[:10]
for idx in top_negative_idx:
    print(f"{train_returns[idx]:>+10.4f}% {train_labels[idx]:<8} {data['label_names'][train_labels[idx]]}")

print("\n" + "="*80)
print("SEQUENCE VISUALIZATION (First Sample)")
print("="*80)

# Show the full price sequence for one sample
sample_idx = 0
sample_prices = X_train[sample_idx, :, 3]  # Close prices for all 128 days
sample_label = y_train[sample_idx]

print(f"\nSample {sample_idx}:")
print(f"  Label: {sample_label} ({data['label_names'][sample_label]})")
print(f"  Day 127 close: {sample_prices[126]:.5f}")
print(f"  Day 128 close: {sample_prices[127]:.5f}")
print(f"  Return: {(sample_prices[127] - sample_prices[126]) / sample_prices[126] * 100:+.4f}%")

print(f"\n  Last 10 days of price sequence:")
for i in range(118, 128):
    print(f"    Day {i+1:3d}: {sample_prices[i]:.5f}")

print("\n" + "="*80)
print("DONE - Dataset inspection complete")
print("="*80)
