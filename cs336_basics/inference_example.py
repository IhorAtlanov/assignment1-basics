"""
Inference Example for Trained Transformer LM

This script demonstrates how to:
1. Load a trained model from checkpoint
2. Generate text using various sampling strategies
3. Evaluate perplexity on test data
4. Interactive generation mode

Usage:
    # Generate text from a prompt
    python inference_example.py \
        --checkpoint checkpoints/best_model.pt \
        --tokenizer bpe_model.pkl \
        --prompt "Once upon a time"
    
    # Calculate perplexity
    python inference_example.py \
        --checkpoint checkpoints/best_model.pt \
        --tokenizer bpe_model.pkl \
        --calculate-perplexity \
        --test-data test.npy
    
    # Interactive mode
    python inference_example.py \
        --checkpoint checkpoints/best_model.pt \
        --tokenizer bpe_model.pkl \
        --interactive
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import pickle

from cs336_basics.TransformerLM.transformer_lm import TransformerLM
from cs336_basics.BPE.Tokenizer import Tokenizer

class TextGenerator:
    def __init__(self, model: TransformerLM, tokenizer: Tokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
        stop_tokens: list = None,
        stream: bool = False
    ):
        """
        Generate text from a prompt
        
        Args:
            prompt: Starting text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens (>1.0 = less repetition)
            stop_tokens: List of token IDs that stop generation
            stream: Whether to yield tokens as they're generated
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            raise ValueError("Prompt encoded to empty sequence")
        
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        generated_ids = input_ids.clone()
        
        if stop_tokens is None:
            stop_tokens = []
        
        for i in range(max_new_tokens):
            # Get context window (last context_length tokens)
            context = generated_ids[:, -self.model.context_length:]
            
            # Forward pass
            logits = self.model(context)
            next_token_logits = logits[:, -1, :].clone()
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[0].tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = False
                # Shift right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                
                # Scatter to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for stop tokens
            if next_token.item() in stop_tokens:
                break
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stream output if requested
            if stream:
                token_text = self.tokenizer.decode([next_token.item()])
                yield token_text
        
        if not stream:
            # Decode entire generated text
            generated_text = self.tokenizer.decode(generated_ids[0].tolist())
            yield generated_text


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str, device: str = 'cuda'):
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Make sure the checkpoint was created by train.py"
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Loading model configuration...")
    print(f"  Vocab size: {config['vocab_size']}")
    print(f"  Context length: {config['context_length']}")
    print(f"  d_model: {config['d_model']}")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Heads: {config['num_heads']}")
    
    # Create model
    device_obj = torch.device(device)
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config.get('d_ff'),
        theta=config.get('theta', 10000.0),
        device=device_obj
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Ensure all model parameters are on the correct device
    model = model.to(device_obj)
    model.eval()
    
    iteration = checkpoint.get('iteration', 'unknown')
    print(f"Loaded model from iteration {iteration}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, 'rb') as f:
        data = pickle.load(f)
    
    tokenizer = Tokenizer(
        vocab=data['vocab'],
        merges=data['merges'],
        special_tokens=data.get('special_tokens', [])
    )
    
    print(f"Tokenizer vocab size: {len(tokenizer.vocab)}")
    
    return model, tokenizer, config

def generate_text_simple(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    repetition_penalty: float = 1.0,
    device: str = 'cuda'
):
    """Simple text generation with nice output formatting"""
    
    generator = TextGenerator(model, tokenizer, device)
    
    print(f"\n{'='*70}")
    print(f"Prompt: {prompt}")
    print(f"{'='*70}")
    print("Settings:")
    print(f"  Max tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    if top_k:
        print(f"  Top-k: {top_k}")
    if top_p:
        print(f"  Top-p: {top_p}")
    if repetition_penalty != 1.0:
        print(f"  Repetition penalty: {repetition_penalty}")
    print(f"{'='*70}\n")
    
    print("Generating...\n")
    
    # Generate text (non-streaming for clean output)
    generated_text = next(generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stream=False
    ))
    
    print(generated_text)
    print(f"\n{'='*70}")
    print(f"Generation complete ({len(tokenizer.encode(generated_text))} tokens total)")
    print(f"{'='*70}\n")
    
    return generated_text


def interactive_mode(model: TransformerLM, tokenizer: Tokenizer, device: str = 'cuda'):
    """Interactive text generation mode"""
    
    generator = TextGenerator(model, tokenizer, device)
    
    print("\n" + "="*70)
    print("Interactive Generation Mode")
    print("="*70)
    print("\nCommands:")
    print("  /quit or /exit - Exit interactive mode")
    print("  /temp <value> - Set temperature (default: 1.0)")
    print("  /topk <value> - Set top-k (default: None)")
    print("  /topp <value> - Set top-p (default: None)")
    print("  /maxtokens <value> - Set max tokens (default: 100)")
    print("  /settings - Show current settings")
    print("\nJust type your prompt and press Enter to generate text.")
    print("="*70 + "\n")
    
    # Default settings
    settings = {
        'temperature': 1.0,
        'top_k': None,
        'top_p': None,
        'max_tokens': 100,
        'repetition_penalty': 1.0
    }
    
    while True:
        try:
            prompt = input(">>> ").strip()
            
            if not prompt:
                continue
            
            # Handle commands
            if prompt.startswith('/'):
                parts = prompt.split()
                cmd = parts[0].lower()
                
                if cmd in ['/quit', '/exit']:
                    print("Exiting interactive mode...")
                    break
                
                elif cmd == '/settings':
                    print("\nCurrent settings:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    print()
                    continue
                
                elif cmd == '/temp' and len(parts) > 1:
                    try:
                        settings['temperature'] = float(parts[1])
                        print(f"Temperature set to {settings['temperature']}")
                    except ValueError:
                        print("Invalid temperature value")
                    continue
                
                elif cmd == '/topk' and len(parts) > 1:
                    try:
                        settings['top_k'] = int(parts[1]) if parts[1].lower() != 'none' else None
                        print(f"Top-k set to {settings['top_k']}")
                    except ValueError:
                        print("Invalid top-k value")
                    continue
                
                elif cmd == '/topp' and len(parts) > 1:
                    try:
                        settings['top_p'] = float(parts[1]) if parts[1].lower() != 'none' else None
                        print(f"Top-p set to {settings['top_p']}")
                    except ValueError:
                        print("Invalid top-p value")
                    continue
                
                elif cmd == '/maxtokens' and len(parts) > 1:
                    try:
                        settings['max_tokens'] = int(parts[1])
                        print(f"Max tokens set to {settings['max_tokens']}")
                    except ValueError:
                        print("Invalid max tokens value")
                    continue
                
                else:
                    print(f"Unknown command: {cmd}")
                    continue
            
            # Generate text
            print("\nGenerating...\n")
            generated_text = next(generator.generate(
                prompt=prompt,
                max_new_tokens=settings['max_tokens'],
                temperature=settings['temperature'],
                top_k=settings['top_k'],
                top_p=settings['top_p'],
                repetition_penalty=settings['repetition_penalty'],
                stream=False
            ))
            
            print(generated_text)
            print("\n" + "-"*70 + "\n")
        
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Inference with trained Transformer LM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to tokenizer pickle file (.pkl)')
    
    # Generation settings
    gen_group = parser.add_argument_group('Generation Settings')
    gen_group.add_argument('--prompt', type=str, default='Once upon a time',
                          help='Text prompt for generation')
    gen_group.add_argument('--max-tokens', type=int, default=100,
                          help='Maximum tokens to generate')
    gen_group.add_argument('--temperature', type=float, default=1.0,
                          help='Sampling temperature (higher = more random)')
    gen_group.add_argument('--top-k', type=int, default=None,
                          help='Top-k sampling (keep only top k tokens)')
    gen_group.add_argument('--top-p', type=float, default=None,
                          help='Top-p/nucleus sampling (cumulative probability threshold)')
    gen_group.add_argument('--repetition-penalty', type=float, default=1.0,
                          help='Repetition penalty (>1.0 = less repetition)')
    
    # Modes
    mode_group = parser.add_argument_group('Modes')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Interactive generation mode')
    
    # Perplexity settings
    ppl_group = parser.add_argument_group('Perplexity Settings')
    ppl_group.add_argument('--test-data', type=str, default=None,
                          help='Path to test data (.npy) for perplexity calculation')
    ppl_group.add_argument('--batch-size', type=int, default=32,
                          help='Batch size for perplexity calculation')
    ppl_group.add_argument('--num-batches', type=int, default=100,
                          help='Number of batches for perplexity calculation')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    print("="*70)
    print("Transformer LM Inference")
    print("="*70 + "\n")
    
    # Check if files exist
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    if not Path(args.tokenizer).exists():
        print(f"ERROR: Tokenizer not found: {args.tokenizer}")
        return
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model and tokenizer
    try:
        model, tokenizer, config = load_model_and_tokenizer(
            args.checkpoint,
            args.tokenizer,
            args.device
        )
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, args.device)
        return
    
    # Text generation (default mode)
    print("\n" + "="*70)
    print("Text Generation")
    print("="*70)
    
    try:
        generate_text_simple(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device
        )
    except Exception as e:
        print(f"ERROR generating text: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()