import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.multiprocessing as mp


if mp.get_start_method(allow_none=True) != 'fork':
    try:
        mp.set_start_method('fork', force=True)
        print("Multiprocessing start method set to 'fork'", flush=True)
    except RuntimeError:
        print("Warning: Unable to set start method to 'fork'", flush=True)

from experiments.runner import exp_ddp
from models.Perceiver import Perceiver
from utils.seeding import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Model Training Parameters')
    
    # Hardware parameters
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use, default is all available GPUs')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master node address for distributed training')
    parser.add_argument('--master_port', type=str, default='12355', help='Master port for distributed training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_epochs', type=int, default=8)  # 8个epoch
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)  # 修改：减少patience
    parser.add_argument('--use_amp', action='store_true') 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of train/validation/test data to use (0.0-1.0, e.g., 1/30 ≈ 0.0333)')
    
    # Model parameters - 修改为新模型的参数
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--seq_size', type=int, default=96, help='Sequence length (should match data)')
    parser.add_argument('--num_features', type=int, default=10, help='Number of input features (already includes factors)')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)

    # 修改：使用新的参数名
    model_config = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "seq_size": args.seq_size,
        "num_features": args.num_features,
        "num_heads": args.num_heads,
        "num_classes": args.num_classes,
        "dropout": args.dropout,
    }

    available_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus if args.num_gpus is not None else available_gpus
    num_gpus = min(num_gpus, available_gpus)

    print(f"Using {num_gpus} GPUs for training", flush=True)

    exp_ddp(
        world_size=num_gpus,
        model_class=Perceiver,
        model_config=model_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        use_amp=args.use_amp,
        master_addr=args.master_addr,
        master_port=args.master_port,
        seed=args.seed,
        data_fraction=args.data_fraction
    )