# forex_atr_by_pair.npz forex_atr_by_time.npz  forex_std_by_pair.npz  forex_std_by_time.npz

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler
from torch.multiprocessing import Process

from dataset.dataprovider import create_dataset, create_dataloader, create_test_dataset
from experiments.solver import trainer, evaluator
from utils.seeding import set_local_seed


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, val_f1):
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1
    }
    torch.save(checkpoint, "checkpoints/best_model.pt")


def exp_rnn(rank, world_size, model_class, model_config, batch_size, num_workers,
            num_epochs, learning_rate, patience, use_amp, master_addr, master_port, seed, data_fraction=1.0):
    try:
        torch.backends.cudnn.benchmark = True
        set_local_seed(seed + rank)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        
        # data_path = "./fin_hloc/forex_atr_by_time.npz"
        data_path = "/home/corelabtq/Desktop/Research/forex/fin_factor/forex_atr_by_time.npz"

        train_dataset, val_dataset, train_size, val_size = create_dataset(
            data_path=data_path, 
            dtype=torch.float32, 
            rank=rank,
            data_fraction=data_fraction
        )
        if rank == 0:
            print(f"Training set size: {train_size}, Validation set size: {val_size}", flush=True)

        train_loader, train_sampler = create_dataloader(
            train_dataset, batch_size, num_workers, world_size, rank, shuffle=True
        )
        val_loader, val_sampler = create_dataloader(
            val_dataset, batch_size, num_workers, world_size, rank, shuffle=False
        )

        # 加载测试集
        test_dataset, test_size = create_test_dataset(
            data_path=data_path,
            dtype=torch.float32,
            rank=rank,
            data_fraction=data_fraction
        )
        
        test_loader = None
        test_sampler = None
        if test_dataset is not None:
            test_loader, test_sampler = create_dataloader(
                test_dataset, batch_size, num_workers, world_size, rank, shuffle=False
            )
            if rank == 0:
                print(f"Test set size: {test_size}", flush=True)

        model = model_class(**model_config).to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

        if rank == 0:
            print(f"Model parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
            for k, v in model_config.items():
                print(f"Model config - {k}: {v}", flush=True)
            print(f"About to start training for {num_epochs} epochs...\n", flush=True)

        scaler = GradScaler(enabled=use_amp)
        
        # 使用 Focal Loss
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        dist.barrier()

        best_val_loss = float('inf')
        best_val_f1 = 0.0
        no_improve_epochs = 0
        early_stop_flag = torch.tensor([0], device=device)
        
        # 联合损失的权重参数
        lambda_return = 0.1   # 回报损失权重
        lambda_sharpe = 0.5   # Sharpe比率损失权重

        # 训练循环
        for epoch in range(num_epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            if test_sampler is not None:
                test_sampler.set_epoch(epoch)

            # ============ 训练 ============
            start_train = time.time()
            train_loss, train_acc, train_precision, train_recall, train_f1_macro, train_f1_weighted = trainer(
                model, train_loader, optimizer, criterion, device, scaler, use_amp, epoch, rank, lambda_return=lambda_return, lambda_sharpe=lambda_sharpe
            )
            train_time = time.time() - start_train

            # ============ 验证 ============
            start_val = time.time()
            val_loss, val_acc, val_precision, val_recall, val_f1_macro, val_f1_weighted = evaluator(
                model, val_loader, criterion, device, use_amp, rank, dataset_name="Validation"
            )
            val_time = time.time() - start_val

            # ============ 测试 ============
            test_loss = 0.0
            test_acc = 0.0
            test_precision = 0.0
            test_recall = 0.0
            test_f1_macro = 0.0
            test_f1_weighted = 0.0
            test_time = 0.0
            
            if test_loader is not None:
                start_test = time.time()
                test_loss, test_acc, test_precision, test_recall, test_f1_macro, test_f1_weighted = evaluator(
                    model, test_loader, criterion, device, use_amp, rank, dataset_name="Test"
                )
                test_time = time.time() - start_test

            # ============ 打印 Epoch 总结 ============
            if rank == 0:
                scheduler.step(val_loss)

                print(f"\n{'='*100}", flush=True)
                print(f"EPOCH {epoch+1}/{num_epochs} SUMMARY", flush=True)
                print(f"{'='*100}", flush=True)
                print(f"{'Dataset':<12} {'Loss':>10} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Time':>8}", flush=True)
                print(f"{'-'*100}", flush=True)
                print(f"{'Training':<12} {train_loss:>10.6f} {train_acc:>7.2f}% {train_precision:>7.2f}% "
                      f"{train_recall:>7.2f}% {train_f1_macro:>7.2f}% {train_time:>7.1f}s", flush=True)
                print(f"{'Validation':<12} {val_loss:>10.6f} {val_acc:>7.2f}% {val_precision:>7.2f}% "
                      f"{val_recall:>7.2f}% {val_f1_macro:>7.2f}% {val_time:>7.1f}s", flush=True)
                if test_loader is not None:
                    print(f"{'Test':<12} {test_loss:>10.6f} {test_acc:>7.2f}% {test_precision:>7.2f}% "
                          f"{test_recall:>7.2f}% {test_f1_macro:>7.2f}% {test_time:>7.1f}s", flush=True)
                print(f"{'-'*100}", flush=True)
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}", flush=True)
                print(f"{'='*100}\n", flush=True)

                # 保存最佳模型
                if val_f1_macro > best_val_f1:
                    best_val_f1 = val_f1_macro
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, epoch, val_loss, val_acc, val_f1_macro)
                    print(f"✓ New best model saved! Val F1: {best_val_f1:.2f}%, Val Loss: {best_val_loss:.6f}\n", flush=True)
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    print(f"No improvement for {no_improve_epochs} epochs (Best Val F1: {best_val_f1:.2f}%)\n", flush=True)

                # Early stopping
                if no_improve_epochs >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}", flush=True)
                    print(f"Best validation F1: {best_val_f1:.2f}%\n", flush=True)
                    early_stop_flag[0] = 1 

            dist.broadcast(early_stop_flag, src=0)

            if early_stop_flag.item() == 1:
                break

            dist.barrier()

        # ============ 训练完成 ============
        if rank == 0:
            print(f"\n{'='*100}", flush=True)
            print(f"TRAINING COMPLETED", flush=True)
            print(f"{'='*100}", flush=True)
            print(f"Best Validation F1: {best_val_f1:.2f}%", flush=True)
            print(f"Best Validation Loss: {best_val_loss:.6f}", flush=True)
            print(f"{'='*100}\n", flush=True)

    except Exception as e:
        if rank == 0:
            print(f"Error in process {rank}: {str(e)}", flush=True)
        raise  
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def exp_ddp(world_size, model_class, model_config, batch_size, num_workers,
            num_epochs, learning_rate, patience, use_amp, master_addr, master_port, seed, data_fraction=1.0):
    processes = []
    for rank in range(world_size):
        p = Process(target=exp_rnn, args=(
            rank, world_size, model_class, model_config, batch_size,
            num_workers, num_epochs, learning_rate, patience,
            use_amp, master_addr, master_port, seed, data_fraction
        ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()