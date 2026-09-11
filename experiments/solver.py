import torch
import torch.distributed as dist
from torch.amp import autocast
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import time
import numpy as np
import sys
import os

# Add parent directory to path to import metrics
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.metrics import (
    cross_entropy_loss, mcfadden_pseudo_r2, rank_information_coefficient
) 

TRADING_DAYS_PER_YEAR = 252
TRADE_CONFIDENCE_THRESHOLD = 0.5


def aggregate_returns_by_time(returns, time_ids=None):
    """Sort chronologically and equal-weight simultaneous signals."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    valid = np.isfinite(r)
    if time_ids is None:
        return r[valid]
    t = np.asarray(time_ids).reshape(-1)
    if t.size != r.size:
        return r[valid]
    valid &= np.isfinite(t)
    r, t = r[valid], t[valid]
    if r.size == 0:
        return r
    order = np.argsort(t, kind="mergesort")
    r, t = r[order], t[order]
    _, inverse = np.unique(t, return_inverse=True)
    return np.bincount(inverse, weights=r) / np.maximum(np.bincount(inverse), 1)


def aggregate_executed_returns_by_time(returns, executed_mask, time_ids=None):
    """Equal-weight executed positions each day; no-trade days earn exactly zero."""
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    executed = np.asarray(executed_mask, dtype=bool).reshape(-1)
    if time_ids is None:
        return r[executed & np.isfinite(r)]
    t = np.asarray(time_ids).reshape(-1)
    valid = np.isfinite(r) & np.isfinite(t)
    r, executed, t = r[valid], executed[valid], t[valid]
    order = np.argsort(t, kind="mergesort")
    r, executed, t = r[order], executed[order], t[order]
    _, inverse = np.unique(t, return_inverse=True)
    sums = np.bincount(inverse, weights=np.where(executed, r, 0.0))
    counts = np.bincount(inverse, weights=executed.astype(np.float64))
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def compounded_cumulative_return(returns, time_ids=None):
    """Compound decimal period returns into a total return."""
    r = aggregate_returns_by_time(returns, time_ids)
    if r.size == 0:
        return 0.0
    if np.any(r <= -1.0):
        return -1.0
    return float(np.expm1(np.sum(np.log1p(r))))


def compounded_max_drawdown(returns, time_ids=None, aggregate_by_time=True):
    """
    用复利净值曲线计算最大回撤：
      equity_t = cumprod(1 + return_t)

    参数 returns 使用小数收益率（例如 1% -> 0.01）。
    若提供 time_ids，则先按时间排序；当 aggregate_by_time=True 时，
    同一 time_id 的收益先做均值聚合，再计算回撤。
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    if r.size <= 1:
        return 0.0

    valid_mask = np.isfinite(r)
    if time_ids is not None:
        t = np.asarray(time_ids).reshape(-1)
        if t.size == r.size:
            valid_mask = valid_mask & np.isfinite(t)
            r = r[valid_mask]
            t = t[valid_mask]

            if r.size <= 1:
                return 0.0

            order = np.argsort(t, kind='mergesort')
            r = r[order]
            t = t[order]

            if aggregate_by_time:
                uniq_t, inverse = np.unique(t, return_inverse=True)
                sums = np.bincount(inverse, weights=r)
                counts = np.bincount(inverse)
                r = sums / np.maximum(counts, 1)
        else:
            r = r[valid_mask]
    else:
        r = r[valid_mask]

    if r.size <= 1:
        return 0.0

    equity = np.cumprod(1.0 + r)
    # Include the initial equity of 1 so an immediate loss is counted as DD.
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    drawdowns = (equity - running_peak) / np.maximum(running_peak, 1e-12)
    return float(abs(np.min(drawdowns)))


def _sanitize_returns(returns):
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    return r[np.isfinite(r)]


def compounded_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=TRADING_DAYS_PER_YEAR, annualize=True):
    """Sharpe calculated from compound-compatible log returns."""
    r = _sanitize_returns(returns)
    if r.size <= 1:
        return 0.0

    log_returns = np.log1p(np.maximum(r, -1.0 + 1e-15))
    rf_per_period = np.log1p(risk_free_rate) / periods_per_year
    excess = log_returns - rf_per_period
    std_excess = np.std(excess, ddof=1)
    if std_excess <= 1e-12:
        return 0.0

    sr = np.mean(excess) / std_excess
    if annualize:
        sr = sr * np.sqrt(periods_per_year)
    return float(sr)


def compounded_annualized_return(returns, periods_per_year=TRADING_DAYS_PER_YEAR):
    """
    Compound annualized return:
      (prod(1 + r_period)) ** (P / N) - 1
    """
    r = _sanitize_returns(returns)
    if r.size == 0:
        return 0.0
    if np.any(r <= -1.0):
        return -1.0
    return float(np.expm1(np.sum(np.log1p(r)) * periods_per_year / r.size))


def compounded_annualized_volatility(returns, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Annualized volatility of log returns."""
    r = _sanitize_returns(returns)
    if r.size <= 1:
        return 0.0
    log_returns = np.log1p(np.maximum(r, -1.0 + 1e-15))
    return float(np.std(log_returns, ddof=1) * np.sqrt(periods_per_year))



def trainer(model, train_loader, optimizer, criterion, device, scaler, use_amp, epoch, rank, lambda_return=0.05, lambda_sharpe=0.05):
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    all_predictions = []
    all_labels = []

    if rank == 0:
        print(f"\n[Epoch {epoch+1}] Starting training, total batches: {len(train_loader)}", flush=True)

    for batch_idx, batch in enumerate(train_loader):
        start_time = time.time()

        inputs, labels, target_prices = batch[:3]

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target_prices = target_prices.to(device, non_blocking=True)
        batch_size = inputs.size(0)
        total_samples += batch_size

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item() * batch_size
        
        # 计算准确率
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        
        # 收集预测和标签用于计算F1
        all_predictions.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
        duration = time.time() - start_time

        # 修改：打印 running average 而不是单个 batch
        if rank == 0 and (batch_idx + 1) % 100 == 0:
            running_acc = (total_correct / total_samples) * 100
            running_loss = total_loss / total_samples
            print(f"[Train] Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                  f"Running Loss: {running_loss:.6f}, Running Acc: {running_acc:.2f}%, Time: {duration:.2f}s", flush=True)

    # 计算平均loss和准确率
    metrics = torch.tensor([total_loss, total_samples, total_correct], device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    loss_avg = metrics[0].item() / metrics[1].item()
    acc_avg = metrics[2].item() / metrics[1].item() * 100
    
    # 计算详细指标（只在rank 0计算）
    f1_macro = 0.0
    f1_weighted = 0.0
    precision_macro = 0.0
    recall_macro = 0.0
    
    if rank == 0:
        # 计算所有指标
        precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
        recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
        f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0) * 100
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        print(f"\n[Epoch {epoch+1}] Training Summary:", flush=True)
        print(f"  Loss: {loss_avg:.6f}", flush=True)
        print(f"  Accuracy: {acc_avg:.2f}%", flush=True)
        print(f"  Precision (Macro): {precision_macro:.2f}%", flush=True)
        print(f"  Recall (Macro): {recall_macro:.2f}%", flush=True)
        print(f"  F1 (Macro): {f1_macro:.2f}%", flush=True)
        print(f"  F1 (Weighted): {f1_weighted:.2f}%", flush=True)
        print(f"  F1 per class - Down: {f1_per_class[0]:.2f}%, Stable: {f1_per_class[1]:.2f}%, Up: {f1_per_class[2]:.2f}%", flush=True)
        print(f"\n  Confusion Matrix:", flush=True)
        print(f"              Pred Down  Pred Stable  Pred Up", flush=True)
        print(f"  True Down      {conf_matrix[0,0]:7d}     {conf_matrix[0,1]:7d}   {conf_matrix[0,2]:7d}", flush=True)
        print(f"  True Stable    {conf_matrix[1,0]:7d}     {conf_matrix[1,1]:7d}   {conf_matrix[1,2]:7d}", flush=True)
        print(f"  True Up        {conf_matrix[2,0]:7d}     {conf_matrix[2,1]:7d}   {conf_matrix[2,2]:7d}", flush=True)

    # 广播指标到所有进程
    metrics_tensor = torch.tensor([loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted], device=device)
    if dist.is_initialized():
        dist.broadcast(metrics_tensor, src=0)
    
    # 解包指标
    loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted = metrics_tensor.tolist()

    return loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted



def evaluator(model, val_loader, criterion, device, use_amp, rank=0,
              dataset_name="Validation"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    
    # 收集概率、标签和目标价格数据
    all_probs = [] 
    all_labels = []
    all_day128_close = []  # 第128天收盘价 (输入序列最后一天)
    all_day129_close = []  # 第129天收盘价 (目标日)
    all_target_returns = []  # schema-safe close/pre_close - 1
    all_pair_names = []

    if rank == 0:
        print(f"\n[{dataset_name}] Starting evaluation, total batches: {len(val_loader)}", flush=True)

    with torch.no_grad():
        all_time_ids = []  # 样本对应的时间桶（用于按天聚合）

        for batch_idx, batch in enumerate(val_loader):
            start_time = time.time()

            # 支持 dataset 返回:
            # (x, y, target), (x, y, target, time_id), (x, y, target, time_id, pair_name)
            if len(batch) == 6:
                inputs, labels, target_prices, time_ids, pair_names, target_returns = batch
            elif len(batch) == 5:
                inputs, labels, target_prices, time_ids, target_returns = batch
                pair_names = None
            else:
                raise ValueError("Evaluation batches must include schema-safe target returns")

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            target_prices = target_prices.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            total_samples += batch_size

            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * batch_size

            # 计算标准准确率 (Argmax, 默认阈值逻辑)
            # 获取概率分布
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            
            # 收集评估所需数据
            # inputs shape: [batch, 128, 10] where feature index 3 is close price
            # target_prices shape: [batch, 10] where index 3 is close price
            all_probs.extend(probs.float().cpu().numpy())
            all_labels.extend(labels.cpu().tolist())
            # 第128天的收盘价 (输入序列的最后一天)
            all_day128_close.extend(inputs[:, -1, 3].float().cpu().numpy())
            # 第129天的收盘价 (目标日)
            all_day129_close.extend(target_prices[:, 3].float().cpu().numpy())
            all_target_returns.extend(target_returns.float().cpu().numpy())
            if time_ids is not None:
                all_time_ids.extend(time_ids.cpu().tolist())
            if pair_names is not None:
                if isinstance(pair_names, (list, tuple)):
                    all_pair_names.extend([str(x) for x in pair_names])
                else:
                    all_pair_names.extend([str(x) for x in pair_names.cpu().tolist()])
            
            duration = time.time() - start_time

            if rank == 0 and (batch_idx + 1) % 50 == 0:
                running_acc = (total_correct / total_samples) * 100
                running_loss = total_loss / total_samples
                print(f"[{dataset_name}] Batch {batch_idx+1}/{len(val_loader)}, "
                      f"Running Loss: {running_loss:.6f}, Running Acc: {running_acc:.2f}%, Time: {duration:.2f}s", flush=True)

    # 同步基础 Loss 和 Accuracy 指标 (所有 GPU)
    metrics = torch.tensor([total_loss, total_samples, total_correct], device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    loss_avg = metrics[0].item() / metrics[1].item()
    acc_avg = metrics[2].item() / metrics[1].item() * 100
    
    # 预初始化指标变量
    f1_macro = 0.0
    f1_weighted = 0.0
    precision_macro = 0.0
    recall_macro = 0.0
    
    if rank == 0:
        # 将收集到的数据转为 Numpy 数组方便计算
        np_probs = np.array(all_probs)           # Shape: [N, 3]
        np_labels = np.array(all_labels)         # Shape: [N]
        np_day128_close = np.array(all_day128_close)  # Shape: [N]
        np_day129_close = np.array(all_day129_close)  # Shape: [N]
        actual_returns = np.asarray(all_target_returns, dtype=np.float64)
        np_time_ids = np.array(all_time_ids) if len(all_time_ids) == len(all_labels) and len(all_time_ids) > 0 else None
        np_pair_names = np.array(all_pair_names, dtype=object) if len(all_pair_names) == len(all_labels) and len(all_pair_names) > 0 else None
        np_pair_names_norm = None
        if np_pair_names is not None:
            np_pair_names_norm = np.char.replace(np.char.upper(np_pair_names.astype(str)), '/', '')

        # 原始的基础预测 (Argmax)
        np_preds = np.argmax(np_probs, axis=1)

        # 计算标准指标
        precision_macro = precision_score(np_labels, np_preds, average='macro', zero_division=0) * 100
        recall_macro = recall_score(np_labels, np_preds, average='macro', zero_division=0) * 100
        f1_macro = f1_score(np_labels, np_preds, average='macro', zero_division=0) * 100
        f1_weighted = f1_score(np_labels, np_preds, average='weighted', zero_division=0) * 100
        f1_per_class = f1_score(np_labels, np_preds, average=None, zero_division=0) * 100
        conf_matrix = confusion_matrix(np_labels, np_preds)
        
        print(f"\n[{dataset_name}] Standard Summary (Argmax):", flush=True)
        print(f"  Loss: {loss_avg:.6f}", flush=True)
        print(f"  Accuracy: {acc_avg:.2f}%", flush=True)
        print(f"  F1 (Macro): {f1_macro:.2f}% | F1 (Weighted): {f1_weighted:.2f}%", flush=True)
        print(f"  F1 per class - Down: {f1_per_class[0]:.2f}%, Stable: {f1_per_class[1]:.2f}%, Up: {f1_per_class[2]:.2f}%", flush=True)
        
        print(f"\n  Confusion Matrix:", flush=True)
        print(f"              Pred Down  Pred Stable  Pred Up", flush=True)
        print(f"  True Down      {conf_matrix[0,0]:7d}     {conf_matrix[0,1]:7d}   {conf_matrix[0,2]:7d}", flush=True)
        print(f"  True Stable    {conf_matrix[1,0]:7d}     {conf_matrix[1,1]:7d}   {conf_matrix[1,2]:7d}", flush=True)
        print(f"  True Up        {conf_matrix[2,0]:7d}     {conf_matrix[2,1]:7d}   {conf_matrix[2,2]:7d}", flush=True)

        # 过滤极小价格样本，避免收益率分母过小导致异常超大百分比
        min_valid_price = 1e-4
        # 使用目标日开盘价作为收益分母，过滤过小的开盘价样本
        valid_price_mask = (np_day128_close >= min_valid_price)
        removed_count = int(np.sum(~valid_price_mask))
        kept_count = int(np.sum(valid_price_mask))

        if removed_count > 0:
            print(
                f"\n  Return-Metric Filter: removed {removed_count} samples with day128_close < {min_valid_price} "
                f"({removed_count / len(np_day128_close) * 100:.2f}%), kept {kept_count}",
                flush=True
            )

        np_probs = np_probs[valid_price_mask]
        np_labels = np_labels[valid_price_mask]
        np_day128_close = np_day128_close[valid_price_mask]
        np_day129_close = np_day129_close[valid_price_mask]
        actual_returns = actual_returns[valid_price_mask]
        if np_time_ids is not None:
            np_time_ids = np_time_ids[valid_price_mask]
        if np_pair_names_norm is not None:
            np_pair_names_norm = np_pair_names_norm[valid_price_mask]

        if len(np_probs) <= 1:
            print(f"\n[{dataset_name}] Return-based metrics may be unstable: insufficient samples after price filtering", flush=True)

        # Return series and execution-cost setup for comprehensive metrics.
        # SCHEMA.txt requires close/pre_close - 1 for these unadjusted stock prices.
        actual_return_pct = actual_returns * 100.0
        max_abs_return_pct = 200.0
        valid_return_mask = np.isfinite(actual_return_pct) & (np.abs(actual_return_pct) <= max_abs_return_pct)
        outlier_removed = int(np.sum(~valid_return_mask))
        if outlier_removed > 0:
            print(
                f"\n  Return Outlier Filter: removed {outlier_removed} samples with |return| > {max_abs_return_pct:.1f}% "
                f"({outlier_removed / len(actual_return_pct) * 100:.2f}%)",
                flush=True,
            )
            np_probs = np_probs[valid_return_mask]
            np_labels = np_labels[valid_return_mask]
            np_day128_close = np_day128_close[valid_return_mask]
            np_day129_close = np_day129_close[valid_return_mask]
            actual_return_pct = actual_return_pct[valid_return_mask]
            if np_time_ids is not None:
                np_time_ids = np_time_ids[valid_return_mask]
            if np_pair_names_norm is not None:
                np_pair_names_norm = np_pair_names_norm[valid_return_mask]

        risk_free_annual = 0.02
        up_prob = np_probs[:, 2]
        down_prob = np_probs[:, 0]
        up_side_returns = actual_return_pct
        down_side_returns = -actual_return_pct

        # ==========================================================
        # Additional Comprehensive Metrics
        # ==========================================================
        print(f"\n[{dataset_name}] Comprehensive Metrics:", flush=True)
        print(f"{'-'*80}")
        
        # Cross-entropy loss and McFadden's Pseudo R²
        ce_loss = cross_entropy_loss(np_labels, np_probs[:, 2])  # Using UP class probability
        pseudo_r2 = mcfadden_pseudo_r2(np_labels == 2, np_probs[:, 2])  # Binary: is UP or not
        
        print(f"  Statistical Metrics:", flush=True)
        print(f"    Cross-Entropy Loss (UP class): {ce_loss:.6f}", flush=True)
        print(f"    McFadden's Pseudo R² (UP prediction): {pseudo_r2:.6f}", flush=True)

        # Rank Information Coefficient (trend score vs realized next-step return)
        trend_score = up_prob - down_prob
        rank_ic = rank_information_coefficient(trend_score, actual_return_pct)
        print(f"    Rank IC (trend score vs actual return): {rank_ic:+.6f}", flush=True)
        
        # Trade directional predictions only above the confidence gate.
        all_pred_classes = np.argmax(np_probs, axis=1)
        all_strategy_returns = np.zeros_like(up_side_returns)
        
        # UP predictions: long position
        up_pred_mask = (all_pred_classes == 2) & (up_prob > TRADE_CONFIDENCE_THRESHOLD)
        all_strategy_returns[up_pred_mask] = up_side_returns[up_pred_mask]
        
        # DOWN predictions: short position
        down_pred_mask = (all_pred_classes == 0) & (down_prob > TRADE_CONFIDENCE_THRESHOLD)
        all_strategy_returns[down_pred_mask] = down_side_returns[down_pred_mask]
        
        executed_trade_mask = up_pred_mask | down_pred_mask
        strategy_time_ids = np_time_ids if np_time_ids is not None and len(np_time_ids) == len(all_strategy_returns) else None
        daily_returns = aggregate_executed_returns_by_time(
            all_strategy_returns, executed_trade_mask, strategy_time_ids
        )

        if len(daily_returns) > 1:
            full_sharpe = compounded_sharpe_ratio(
                daily_returns / 100,
                risk_free_rate=risk_free_annual,
                periods_per_year=TRADING_DAYS_PER_YEAR
            )
            full_ann_return = compounded_annualized_return(daily_returns / 100, periods_per_year=TRADING_DAYS_PER_YEAR)
            full_ann_vol = compounded_annualized_volatility(daily_returns / 100, periods_per_year=TRADING_DAYS_PER_YEAR)
            full_max_dd = compounded_max_drawdown(daily_returns / 100, aggregate_by_time=False)
            full_cum_return = compounded_cumulative_return(daily_returns / 100)
            
            print(f"\n  Confidence-Gated Strategy Portfolio (p > {TRADE_CONFIDENCE_THRESHOLD:.2f}):", flush=True)
            print(f"    Sharpe Ratio (Annualized): {full_sharpe:.4f}", flush=True)
            print(f"    Annualized Return: {full_ann_return*100:.2f}%", flush=True)
            print(f"    Annualized Volatility: {full_ann_vol*100:.2f}%", flush=True)
            print(f"    Maximum Drawdown: {full_max_dd*100:.2f}%", flush=True)
            print(f"    Compounded Cumulative Return: {full_cum_return*100:.2f}%", flush=True)
            print(f"    Trading Days: {len(daily_returns)}", flush=True)
            print(f"    Executed Trades: {int(np.sum(executed_trade_mask)):,}", flush=True)

            executed_trade_returns = all_strategy_returns[executed_trade_mask]
            if len(executed_trade_returns) > 0:
                trade_win_rate = (np.sum(executed_trade_returns > 0) / len(executed_trade_returns)) * 100
                participation_rate = (len(executed_trade_returns) / len(all_strategy_returns)) * 100
                print(f"    Mean Return per Trade: {np.mean(executed_trade_returns):.4f}%", flush=True)
                print(f"    Win Rate (Executed Trades Only): {trade_win_rate:.2f}%", flush=True)
                print(f"    Participation Rate (confidence-gated): {participation_rate:.2f}%", flush=True)
            else:
                print(f"    Win Rate (Executed Trades Only): N/A", flush=True)

        print(f"{'-'*80}")

    # 广播基础指标到所有进程 (保持流程完整性)
    metrics_tensor = torch.tensor([loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted], device=device)
    if dist.is_initialized():
        dist.broadcast(metrics_tensor, src=0)
    
    # 解包
    loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted = metrics_tensor.tolist()
    
    return loss_avg, acc_avg, precision_macro, recall_macro, f1_macro, f1_weighted, loss_avg
