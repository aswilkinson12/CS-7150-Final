"""
TrajectoryGPT - GPT Decoder Implementation
Decoder-only transformer for trajectory prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TrajectoryGPTConfig:
    """Configuration for TrajectoryGPT model"""

    def __init__(
        self,
        vocab_size: int = 1048,
        n_embd: int = 384,
        n_layer: int = 6,
        n_head: int = 6,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        bias: bool = True
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.bias = bias


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention"""

    def __init__(self, config: TrajectoryGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Causal mask to ensure attention only to past
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                 .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v

        # Reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network"""

    def __init__(self, config: TrajectoryGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config: TrajectoryGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TrajectoryGPT(nn.Module):
    """
    GPT Decoder for Trajectory Prediction

    Architecture:
    - Token embeddings + positional embeddings
    - N transformer blocks with causal self-attention
    - Language modeling head for next token prediction
    """

    def __init__(self, config: TrajectoryGPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe = nn.Embedding(config.max_seq_len, config.n_embd),  # Position embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing between token embedding and lm_head (like GPT-2)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print(f"TrajectoryGPT initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self) -> int:
        """Return total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            idx: Token indices (B, T)
            targets: Target token indices (B, T) for training

        Returns:
            logits: (B, T, vocab_size)
            loss: Scalar loss if targets provided, else None
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_seq_len, f"Sequence length {t} exceeds max {self.config.max_seq_len}"

        # Get positions
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (T,)

        # Forward through embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        # Get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        end_token: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively

        Args:
            idx: Starting tokens (B, T)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if not None)
            top_p: Nucleus sampling threshold (if not None)
            end_token: Stop token ID

        Returns:
            Generated token sequence (B, T+max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Optional: top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Optional: nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if end_token is not None and (idx_next == end_token).all():
                break

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop if end token generated
            if (idx_next == end_token).any():
                break

        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str
    ):
        """
        Configure optimizer with weight decay (AdamW style)

        Separate parameters that should/shouldn't be decayed:
        - Apply decay to all weights in matmuls + embeddings
        - Don't apply to biases and layernorm weights
        """
        # Get all parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Create optim groups: decay vs no_decay
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Optimizer: {len(decay_params)} tensors decayed, {len(nodecay_params)} not decayed")
        print(f"  Decayed params: {num_decay_params:,}")
        print(f"  Non-decayed params: {num_nodecay_params:,}")

        # Create AdamW optimizer
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )

        return optimizer


def test_model():
    """Test TrajectoryGPT implementation"""
    print("Testing TrajectoryGPT...")

    # Create config
    config = TrajectoryGPTConfig(
        vocab_size=280,
        n_embd=384,
        n_layer=6,
        n_head=6,
        max_seq_len=64
    )

    # Create model
    model = TrajectoryGPT(config)

    # Test forward pass
    batch_size = 8
    seq_len = 10
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets)

    print(f"✓ Input shape: {idx.shape}")
    print(f"✓ Output logits shape: {logits.shape}")
    print(f"✓ Loss: {loss.item():.4f}")

    # Test generation
    start_tokens = torch.tensor([[257, 136, 137, 138]])  # [START] + history
    generated = model.generate(
        start_tokens,
        max_new_tokens=6,
        temperature=1.0,
        top_p=0.9
    )

    print(f"✓ Generated sequence shape: {generated.shape}")
    print(f"✓ Generated tokens: {generated[0].tolist()}")

    print("\n✓ TrajectoryGPT test passed!")


if __name__ == "__main__":
    test_model()
