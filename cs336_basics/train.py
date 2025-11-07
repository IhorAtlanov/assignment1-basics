import argparse
import os
import time
import numpy as np
import torch
import json

# Import model components
from cs336_basics.TransformerLM.transformer_lm import TransformerLM
from cs336_basics.Cross_entropy_loss_AdamW.adamw import AdamW
from cs336_basics.Cross_entropy_loss_AdamW.cross_entropy import run_cross_entropy
from cs336_basics.Cross_entropy_loss_AdamW.gradient_clipping import clip_grad_l2_
from cs336_basics.Cross_entropy_loss_AdamW.learning_rate_schedule import run_get_lr_cosine_schedule
from cs336_basics.data_loading_checkpointing.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.data_loading_checkpointing.data_loading import get_batch


class TrainingConfig:
    """Configuration for training run"""
    def __init__(self, **kwargs):
        # Model hyperparameters
        self.vocab_size: int = kwargs.get('vocab_size', 10000)
        self.context_length: int = kwargs.get('context_length', 256)
        self.d_model: int = kwargs.get('d_model', 512)
        self.num_layers: int = kwargs.get('num_layers', 4)
        self.num_heads: int = kwargs.get('num_heads', 16)
        self.d_ff: int = kwargs.get('d_ff', 1344)  # Will default to ~8/3 * d_model
        self.theta: float = kwargs.get('theta', 10000.0)
        self.eps: float = kwargs.get('eps', 1e-5)
        
        # Optimizer hyperparameters
        self.learning_rate: float = kwargs.get('learning_rate', 3e-4)
        self.min_learning_rate: float = kwargs.get('min_learning_rate', 3e-5)
        self.weight_decay: float = kwargs.get('weight_decay', 0.1)
        self.beta1: float = kwargs.get('beta1', 0.9)
        self.beta2: float = kwargs.get('beta2', 0.95)
        self.grad_clip: float = kwargs.get('grad_clip', 1.0)
        
        # Learning rate schedule
        self.warmup_iters: int = kwargs.get('warmup_iters', 2000)
        self.cosine_cycle_iters: int = kwargs.get('cosine_cycle_iters', 100000)
        
        # Training hyperparameters
        self.batch_size: int = kwargs.get('batch_size', 64)
        self.max_iters: int = kwargs.get('max_iters', 100000)
        self.eval_interval: int = kwargs.get('eval_interval', 500)
        self.eval_iters: int = kwargs.get('eval_iters', 100)
        self.log_interval: int = kwargs.get('log_interval', 100)
        self.checkpoint_interval: int = kwargs.get('checkpoint_interval', 5000)
        
        # Data paths
        self.train_data_path: str = kwargs.get('train_data_path', 'train.npy')
        self.val_data_path: str = kwargs.get('val_data_path', 'val.npy')
        self.checkpoint_dir: str = kwargs.get('checkpoint_dir', './checkpoints')
        self.resume_from: str | None = kwargs.get('resume_from', None)
        
        # Device and precision
        self.device: str = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype: str = kwargs.get('dtype', 'float32')
        self.compile_model: bool = kwargs.get('compile_model', False)
        
        # Logging
        self.use_wandb: bool = kwargs.get('use_wandb', False)
        self.wandb_project: str = kwargs.get('wandb_project', 'transformer-lm')
        self.wandb_run_name: str | None = kwargs.get('wandb_run_name', None)
        
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class Trainer:
    """Trainer class for Transformer LM"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # history of evals: list of {"iter": int, "train_loss": float, "val_loss": float, "time": float}
        self.eval_history = []
        
        # Normalize device so 'cuda' -> 'cuda:{current_device}', avoiding comparisons like cuda != cuda:0
        if config.device.startswith('cuda') and config.device != 'cpu':
            # if user passed 'cuda' or 'cuda:None', map to actual current device index
            if config.device == 'cuda':
                current = torch.cuda.current_device() if torch.cuda.is_available() else None
                device_str = f'cuda:{current}' if current is not None else 'cpu'
            else:
                device_str = config.device  # e.g. 'cuda:1'
        else:
            device_str = config.device
            
        self.device = torch.device(device_str)
        self.dtype = getattr(torch, config.dtype)
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Save config
        config.save(os.path.join(config.checkpoint_dir, 'config.json'))
        
        # Initialize model
        print("Initializing model...")
        
        # First create on CPU to avoid partial device allocation issues
        self.model = TransformerLM(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            eps=config.eps,
            theta=config.theta,
            device=None,  # Create on default device first
            dtype=self.dtype
        )
        
        # Now move everything to target device
        print(f"Moving model to {self.device}...")
        self.model = self.model.to(self.device)
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model initialized with {self.num_params:,} parameters")
        
        # Verify all parameters and buffers are on correct device
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                print(f"WARNING: Parameter {name} is on {param.device}, expected {self.device}")
        
        for name, buffer in self.model.named_buffers():
            if buffer.device != self.device:
                print(f"WARNING: Buffer {name} is on {buffer.device}, expected {self.device}")
        
        # Compile model if requested (PyTorch 2.0+)
        if config.compile_model:
            try:
                print("Compiling model...")
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        # Load data with memory mapping
        print(f"Loading training data from {config.train_data_path}...")
        self.train_data = np.load(config.train_data_path, mmap_mode='r')
        print(f"Training data shape: {self.train_data.shape}")
        
        print(f"Loading validation data from {config.val_data_path}...")
        self.val_data = np.load(config.val_data_path, mmap_mode='r')
        print(f"Validation data shape: {self.val_data.shape}")
        
        # Training state
        self.iter_num = 0
        self.best_val_loss = float('inf')
        
        # Resume from checkpoint if specified
        if config.resume_from:
            print(f"Resuming from checkpoint: {config.resume_from}")
            self.iter_num = load_checkpoint(
                config.resume_from,
                self.model,
                self.optimizer
            )
            print(f"Resumed from iteration {self.iter_num}")
        
        # Initialize Weights & Biases if requested
        self.wandb = None
        if config.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=config.to_dict()
                )
                print("Weights & Biases initialized")
            except ImportError:
                print("Warning: wandb not installed. Install with 'pip install wandb'")
                config.use_wandb = False
    
    def get_lr(self) -> float:
        """Get current learning rate based on schedule"""
        return run_get_lr_cosine_schedule(
            it=self.iter_num,
            max_learning_rate=self.config.learning_rate,
            min_learning_rate=self.config.min_learning_rate,
            warmup_iters=self.config.warmup_iters,
            cosine_cycle_iters=self.config.cosine_cycle_iters
        )
    
    def set_lr(self, lr: float):
        """Set learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    @torch.no_grad()
    def estimate_loss(self) -> dict:
        """Estimate loss on train and val sets"""
        self.model.eval()
        losses = {}
        
        for split in ['train', 'val']:
            data = self.train_data if split == 'train' else self.val_data
            total_loss = 0.0
            
            for _ in range(self.config.eval_iters):
                inputs, targets = get_batch(
                    data,
                    self.config.batch_size,
                    self.config.context_length,
                    device=str(self.device)  # Convert to string for compatibility
                )
                
                logits = self.model(inputs)
                loss = run_cross_entropy(logits, targets)
                total_loss += loss.item()
            
            losses[split] = total_loss / self.config.eval_iters
        
        self.model.train()
        return losses
    
    def train_step(self) -> float:
        """Perform a single training step"""
        # Get learning rate and update optimizer
        lr = self.get_lr()
        self.set_lr(lr)
        
        # Get batch
        inputs, targets = get_batch(
            self.train_data,
            self.config.batch_size,
            self.config.context_length,
            device=str(self.device)  # Convert to string for compatibility
        )
        
        # Forward pass
        logits = self.model(inputs)
        loss = run_cross_entropy(logits, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = clip_grad_l2_(
            self.model.parameters(),
            max_l2_norm=self.config.grad_clip
        )
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item(), grad_norm, lr
    
    def save_checkpoint_file(self, filename: str):
        """Save checkpoint to file"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        save_checkpoint(
            self.model,
            self.optimizer,
            self.iter_num,
            checkpoint_path
        )
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        start_time_glob = time.time()
        print("\n" + "="*70)
        print("Starting training")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")
        print(f"Total iterations: {self.config.max_iters}")
        print(f"Starting from iteration: {self.iter_num}")
        print("="*70 + "\n")
        
        self.model.train()
        start_time = time.time()
        
        while self.iter_num < self.config.max_iters:
            # Training step
            loss, grad_norm, lr = self.train_step()
            self.iter_num += 1
            
            # Logging
            if self.iter_num % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (
                    self.config.batch_size * 
                    self.config.context_length * 
                    self.config.log_interval / 
                    elapsed
                )
                
                losses = self.estimate_loss()
                
                print(
                    f"iter {self.iter_num:6d} | "
                    f"loss {losses['train']:.4f} | "
                    f"val_loss {losses['val']:.4f} | "
                    f"lr {lr:.2e} | "
                    f"grad_norm {grad_norm:.4f} | "
                    f"tokens/sec {tokens_per_sec:.0f}"
                )
                
                if self.config.use_wandb and self.wandb:
                    self.wandb.log({
                        'train/loss': loss,
                        'train/lr': lr,
                        'train/grad_norm': grad_norm,
                        'train/tokens_per_sec': tokens_per_sec,
                        'iter': self.iter_num
                    })
                
                start_time = time.time()
            
            # Evaluation
            if self.iter_num % self.config.eval_interval == 0:
                print("\nEvaluating...")
                losses = self.estimate_loss()
                
                print(
                    f"iter {self.iter_num:6d} | "
                    f"train_loss {losses['train']:.4f} | "
                    f"val_loss {losses['val']:.4f}"
                )
                
                # save eval record
                try:
                    self.eval_history.append({
                        "iter": self.iter_num,
                        "train_loss": float(losses['train']),
                        "val_loss": float(losses['val']),
                        "time": time.time()
                    })
                except Exception:
                    # never crash training because of logging
                    pass

                
                if self.config.use_wandb and self.wandb:
                    self.wandb.log({
                        'eval/train_loss': losses['train'],
                        'eval/val_loss': losses['val'],
                        'iter': self.iter_num
                    })
                
                # Save best model
                if losses['val'] < self.best_val_loss:
                    self.best_val_loss = losses['val']
                    self.save_checkpoint_file('best_model.pt')
                    print(f"New best validation loss: {self.best_val_loss:.4f}")
                
                print()
            
            # Checkpoint saving
            if self.iter_num % self.config.checkpoint_interval == 0:
                self.save_checkpoint_file(f'checkpoint_iter_{self.iter_num}.pt')
        
        print("\n" + "="*70)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*70)
        
        # Save final checkpoint
        self.save_checkpoint_file('final_model.pt')
        
        end_time = time.time()
        
        print(f"Final Time: {(end_time - start_time_glob):.2f} sec.")
        
        # --- after final checkpoint saved ---
        try:
            duration = end_time - getattr(self, '_training_real_start', start_time)

            summary = {
                "learning_rate": float(self.config.learning_rate),
                "min_learning_rate": float(self.config.min_learning_rate),
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(getattr(self, '_training_real_start', start_time_glob))),
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(end_time)),
                "duration_sec": float(duration/60),
                "total_iterations": int(self.iter_num),
                "evals": self.eval_history
            }

            summary_file = "test(lr_and_min_lr).json"
            # read existing list or init new one
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                except Exception:
                    data = []
            else:
                data = []

            data.append(summary)

            with open(summary_file, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"Appended experiment summary to {summary_file}")
        except Exception as e:
            print(f"Warning: could not write summary file: {e}")

        
        if self.config.use_wandb and self.wandb:
            self.wandb.finish()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Transformer Language Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model hyperparameters
    model_group = parser.add_argument_group('Model Hyperparameters')
    model_group.add_argument('--vocab-size', type=int, default=10000,
                            help='Vocabulary size')
    model_group.add_argument('--context-length', type=int, default=256,
                            help='Maximum context length')
    model_group.add_argument('--d-model', type=int, default=512,
                            help='Model dimension')
    model_group.add_argument('--num-layers', type=int, default=6,
                            help='Number of transformer layers')
    model_group.add_argument('--num-heads', type=int, default=8,
                            help='Number of attention heads')
    model_group.add_argument('--d-ff', type=int, default=None,
                            help='Feed-forward dimension (default: ~8/3 * d_model)')
    model_group.add_argument('--theta', type=float, default=10000.0,
                            help='RoPE theta parameter')
    
    # Optimizer hyperparameters
    opt_group = parser.add_argument_group('Optimizer Hyperparameters')
    opt_group.add_argument('--learning-rate', type=float, default=3e-4,
                          help='Maximum learning rate')
    opt_group.add_argument('--min-learning-rate', type=float, default=3e-5,
                          help='Minimum learning rate')
    opt_group.add_argument('--weight-decay', type=float, default=0.1,
                          help='Weight decay coefficient')
    opt_group.add_argument('--beta1', type=float, default=0.9,
                          help='Adam beta1')
    opt_group.add_argument('--beta2', type=float, default=0.95,
                          help='Adam beta2')
    opt_group.add_argument('--grad-clip', type=float, default=1.0,
                          help='Gradient clipping threshold')
    
    # Learning rate schedule
    schedule_group = parser.add_argument_group('Learning Rate Schedule')
    schedule_group.add_argument('--warmup-iters', type=int, default=2000,
                               help='Number of warmup iterations')
    schedule_group.add_argument('--cosine-cycle-iters', type=int, default=100000,
                               help='Iterations for cosine annealing cycle')
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--batch-size', type=int, default=32,
                           help='Batch size')
    train_group.add_argument('--max-iters', type=int, default=100000,
                           help='Maximum number of training iterations')
    train_group.add_argument('--eval-interval', type=int, default=500,
                           help='Evaluation interval')
    train_group.add_argument('--eval-iters', type=int, default=100,
                           help='Number of iterations for evaluation')
    train_group.add_argument('--log-interval', type=int, default=100,
                           help='Logging interval')
    train_group.add_argument('--checkpoint-interval', type=int, default=5000,
                           help='Checkpoint saving interval')
    
    # Data paths
    data_group = parser.add_argument_group('Data Paths')
    data_group.add_argument('--train-data-path', type=str, #required=True,
                          help='Path to training data (.npy file)')
    data_group.add_argument('--val-data-path', type=str, #required=True,
                          help='Path to validation data (.npy file)')
    data_group.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                          help='Directory for saving checkpoints')
    data_group.add_argument('--resume-from', type=str, default=None,
                          help='Path to checkpoint to resume from')
    
    # Device and precision
    device_group = parser.add_argument_group('Device and Precision')
    device_group.add_argument('--device', type=str, default='cuda',
                            choices=['cuda', 'cpu'],
                            help='Device to use for training')
    device_group.add_argument('--dtype', type=str, default='float32',
                            choices=['float32', 'float16', 'bfloat16'],
                            help='Data type for model parameters')
    device_group.add_argument('--compile', action='store_true',
                            help='Compile model with PyTorch 2.0')
    
    # Logging
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--use-wandb', action='store_true',
                         help='Use Weights & Biases for logging')
    log_group.add_argument('--wandb-project', type=str, default='transformer-lm',
                         help='W&B project name')
    log_group.add_argument('--wandb-run-name', type=str, default=None,
                         help='W&B run name')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file (overrides other arguments)')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Load config from file if specified
    if args.config:
        print(f"Loading config from {args.config}")
        config = TrainingConfig.load(args.config)
    else:
        # Create config from command line arguments
        config = TrainingConfig(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            theta=args.theta,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            grad_clip=args.grad_clip,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
            batch_size=args.batch_size,
            max_iters=args.max_iters,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            train_data_path=args.train_data_path,
            val_data_path=args.val_data_path,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume_from,
            device=args.device,
            dtype=args.dtype,
            compile_model=args.compile,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name
        )
    
    # Create trainer and run training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()