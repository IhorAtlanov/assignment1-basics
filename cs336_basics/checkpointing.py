import torch
import os
import typing


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    """
    Save a training checkpoint containing model, optimizer, and iteration state.
    
    Args:
        model: The neural network model to save
        optimizer: The optimizer to save
        iteration: Current training iteration number
        out: File path or file-like object to save the checkpoint to
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    Load a training checkpoint and restore model and optimizer states.
    
    Args:
        src: File path or file-like object to load the checkpoint from
        model: The neural network model to restore state into
        optimizer: The optimizer to restore state into
    
    Returns:
        The iteration number from the checkpoint
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']