from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    from cs336_basics.TransformerLM.liner import Liner
    model = Liner(d_in, d_out, device=weights.device, dtype=weights.dtype)
    
    # Load weights - need to transpose since weights are (d_out, d_in) 
    # but Liner stores W as (d_in, d_out)
    model.load_state_dict({'W': weights.T})
    
    # Run forward pass
    return model.forward(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    from cs336_basics.TransformerLM.embedding import Embedding
    
    # Create an embedding layer with the specified dimensions
    embedding = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
        device=weights.device,
        dtype=weights.dtype
    )
    
    # Load the provided weights into the embedding layer
    with torch.no_grad():
        embedding.weight.copy_(weights)
    
    # Perform the embedding lookup
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    import torch.nn.functional as F
    x_w1 = F.linear(in_features, w1_weight)  # Shape: (..., d_ff)
    silu_output = run_silu(x_w1)  # Shape: (..., d_ff)
    
    # Step 2: Project input through W3 for gating
    x_w3 = F.linear(in_features, w3_weight)  # Shape: (..., d_ff)
    
    # Step 3: Apply gating (element-wise multiplication)
    gated = silu_output * x_w3  # Shape: (..., d_ff)
    
    # Step 4: Project back to d_model through W2
    output = F.linear(gated, w2_weight)  # Shape: (..., d_model)
    
    return output


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    from cs336_basics.TransformerLM.scaled_dot_product_attention import scaled_dot_product_attention
    output, _ = scaled_dot_product_attention(Q, K, V, mask)
    return output


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    from cs336_basics.TransformerLM.multihead_self_attention import MultiHeadSelfAttention
    
    # Create the module
    mha = MultiHeadSelfAttention(d_model, num_heads)
    
    # The weights are already in batched format (d_model x d_in)
    # containing all heads concatenated together
    # Set the projection weights directly
    with torch.no_grad():
        mha.q_proj.weight.copy_(q_proj_weight)
        mha.k_proj.weight.copy_(k_proj_weight)
        mha.v_proj.weight.copy_(v_proj_weight)
        mha.o_proj.weight.copy_(o_proj_weight)
    
    # Run forward pass
    mha.eval()
    with torch.no_grad():
        output = mha(in_features)
    
    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    from cs336_basics.TransformerLM.multihead_self_attention_rope import MultiHeadSelfAttention
    
    mha = MultiHeadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta)
    
    # Set the projection weights
    with torch.no_grad():
        mha.q_proj.weight.copy_(q_proj_weight)
        mha.k_proj.weight.copy_(k_proj_weight)
        mha.v_proj.weight.copy_(v_proj_weight)
        mha.o_proj.weight.copy_(o_proj_weight)
    
    # Run forward pass with RoPE
    mha.eval()
    with torch.no_grad():
        output = mha(in_features, token_positions=token_positions)
    
    return output


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    # basic checks
    if d_k % 2 != 0:
        raise ValueError("d_k must be even for RoPE (pairs of dims).")
    if in_query_or_key.shape[-1] != d_k:
        raise ValueError(f"Last dim of input ({in_query_or_key.shape[-1]}) must equal d_k ({d_k}).")

    device = in_query_or_key.device
    dtype = in_query_or_key.dtype

    # prepare frequency terms: indices 0,2,4,... -> we use d_k/2 frequencies
    indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)  # shape (d_k/2,)
    freqs = 1.0 / (theta ** (indices / d_k))  # shape (d_k/2,)

    # ensure token_positions is float on same device, shape (..., seq_len)
    pos = token_positions.to(device=device)
    pos = pos.to(torch.float32)

    # Broadcast pos[..., seq_len, 1] * freqs[1] -> (..., seq_len, d_k/2)
    # If token_positions was (seq_len,), broadcasting will work automatically.
    freqs_pos = pos.unsqueeze(-1) * freqs.view(*([1] * (pos.dim() - 1)), -1)  # shape (..., seq_len, d_k/2)

    cos = torch.cos(freqs_pos).to(dtype=dtype)
    sin = torch.sin(freqs_pos).to(dtype=dtype)

    # split input into even/odd dims
    x_even = in_query_or_key[..., 0::2]  # (..., seq_len, d_k/2)
    x_odd  = in_query_or_key[..., 1::2]  # (..., seq_len, d_k/2)

    # apply rotation:
    x_even_rotated = x_even * cos - x_odd * sin
    x_odd_rotated  = x_even * sin + x_odd * cos

    # interleave back to (..., seq_len, d_k)
    out = torch.empty_like(in_query_or_key)
    out[..., 0::2] = x_even_rotated
    out[..., 1::2] = x_odd_rotated

    return out


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    from cs336_basics.TransformerLM.transformer_block import TransformerBlock
    
    # Get device and dtype from input
    device = in_features.device
    dtype = in_features.dtype
    
    # Create TransformerBlock with RoPE enabled
    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta
    ).to(device=device, dtype=dtype)
    
    # Load the weights into the model
    # Map from the reference implementation's keys to our implementation's keys
    state_dict = {
        # Multi-head attention weights
        'attention.q_proj.weight': weights['attn.q_proj.weight'],
        'attention.k_proj.weight': weights['attn.k_proj.weight'],
        'attention.v_proj.weight': weights['attn.v_proj.weight'],
        'attention.o_proj.weight': weights['attn.output_proj.weight'],
        
        # First RMSNorm
        'norm1.weight': weights['ln1.weight'],
        
        # Feed-forward network weights
        'feed_forward.w1.weight': weights['ffn.w1.weight'],
        'feed_forward.w2.weight': weights['ffn.w2.weight'],
        'feed_forward.w3.weight': weights['ffn.w3.weight'],
        
        # Second RMSNorm
        'norm2.weight': weights['ln2.weight'],
    }
    
    block.load_state_dict(state_dict, strict=False)
    
    # Set to evaluation mode
    block.eval()
    
    # Create token positions for RoPE (sequential positions)
    batch_size, seq_len, _ = in_features.shape
    token_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Run the forward pass
    with torch.no_grad():
        output = block(in_features, token_positions=token_positions)
    
    return output


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    from cs336_basics.TransformerLM.transformer_lm import TransformerLM
    
    # Get device and dtype from input
    device = in_indices.device
    dtype = next(iter(weights.values())).dtype
    
    # Create TransformerLM with RoPE enabled
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=rope_theta,
        device=device,
        dtype=dtype
    )
    
    # Map from reference implementation's keys to our implementation's keys
    state_dict = {}
    
    # Token embeddings
    state_dict['token_embedding.weight'] = weights['token_embeddings.weight']
    
    # Process each layer
    for layer_idx in range(num_layers):
        layer_prefix = f'layers.{layer_idx}.'
        our_prefix = f'layers.{layer_idx}.'
        
        # Multi-head attention weights
        state_dict[f'{our_prefix}attention.q_proj.weight'] = weights[f'{layer_prefix}attn.q_proj.weight']
        state_dict[f'{our_prefix}attention.k_proj.weight'] = weights[f'{layer_prefix}attn.k_proj.weight']
        state_dict[f'{our_prefix}attention.v_proj.weight'] = weights[f'{layer_prefix}attn.v_proj.weight']
        state_dict[f'{our_prefix}attention.o_proj.weight'] = weights[f'{layer_prefix}attn.output_proj.weight']
        
        # First RMSNorm
        state_dict[f'{our_prefix}norm1.weight'] = weights[f'{layer_prefix}ln1.weight']
        
        # Feed-forward network weights
        state_dict[f'{our_prefix}feed_forward.w1.weight'] = weights[f'{layer_prefix}ffn.w1.weight']
        state_dict[f'{our_prefix}feed_forward.w2.weight'] = weights[f'{layer_prefix}ffn.w2.weight']
        state_dict[f'{our_prefix}feed_forward.w3.weight'] = weights[f'{layer_prefix}ffn.w3.weight']
        
        # Second RMSNorm
        state_dict[f'{our_prefix}norm2.weight'] = weights[f'{layer_prefix}ln2.weight']
    
    # Final layer norm
    state_dict['norm.weight'] = weights['ln_final.weight']
    
    # Output projection (lm_head)
    state_dict['output_proj.W'] = weights['lm_head.weight'].T  # Transpose because Liner uses W not W^T
    
    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    
    # Set to evaluation mode
    model.eval()
    
    # Run the forward pass
    with torch.no_grad():
        logits = model(in_indices)
    
    return logits
    


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    original_dtype = in_features.dtype
    x = in_features.float()
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    x_norm = x_norm.to(original_dtype)
    return x_norm * weights


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    import numpy as np
    # We need context_length + 1 tokens to get both inputs and targets
    max_start_idx = len(dataset) - context_length - 1
    
    if max_start_idx < 0:
        raise ValueError(f"Dataset too small: needs at least {context_length + 1} tokens")
    
    # Sample random starting indices for each sequence in the batch
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # Collect sequences
    inputs = np.array([dataset[i:i + context_length] for i in start_indices])
    targets = np.array([dataset[i + 1:i + context_length + 1] for i in start_indices])
    
    # Convert to PyTorch tensors and move to device
    inputs = torch.from_numpy(inputs.astype(np.int64)).to(device)
    targets = torch.from_numpy(targets.astype(np.int64)).to(device)
    
    return inputs, targets


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    from cs336_basics.TransformerLM.softmax import softmax
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Step 1: Subtract max for numerical stability
    # Subtract the maximum logit value for each example (along vocab dimension)
    max_logits = torch.max(inputs, dim=1, keepdim=True)[0]  # Shape: (batch_size, 1)
    inputs_stable = inputs - max_logits  # Shape: (batch_size, vocab_size)
    
    # Step 2: Compute log(sum(exp(inputs_stable))) for each example
    # This is log_sum_exp along the vocabulary dimension
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_stable), dim=1))  # Shape: (batch_size,)
    
    # Step 3: Extract the logit values at the target indices
    # For each example i, we want inputs_stable[i, targets[i]]
    batch_size = inputs.shape[0]
    target_logits = inputs_stable[torch.arange(batch_size, device=inputs.device), targets]  # Shape: (batch_size,)
    
    # Step 4: Compute cross entropy for each example
    # ℓ_i = -log(softmax(inputs)[target])
    #     = -log(exp(inputs_stable[target]) / sum(exp(inputs_stable)))
    #     = -(inputs_stable[target] - log_sum_exp)
    #     = log_sum_exp - inputs_stable[target]
    cross_entropy_per_example = log_sum_exp - target_logits  # Shape: (batch_size,)
    
    # Step 5: Return the average across the batch
    return cross_entropy_per_example.mean()  # Shape: scalar


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    from cs336_basics.Cross_entropy_loss_AdamW.gradient_clipping import clip_grad_l2_
    return clip_grad_l2_(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    from cs336_basics.Cross_entropy_loss_AdamW.adamw import AdamW
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    from cs336_basics.Cross_entropy_loss_AdamW.learning_rate_schedule import run_get_lr_cosine_schedule
    return run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    from cs336_basics.BPE.Tokenizer import Tokenizer
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    from cs336_basics.BPE.BPE import run_train_bpe
    return run_train_bpe(input_path, vocab_size, special_tokens, **kwargs)
