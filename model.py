import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, d, max_seq_len):
        super().__init__()
        m = torch.arange(max_seq_len) # (max_seq_len, )
        theta = 10000 ** -(torch.arange(0,d,2).float() / d) # (1, d/2)
        angles = torch.outer(m, theta).repeat_interleave(2, dim=-1) # (max_seq_len, d)
        self.register_buffer('angles', angles)
      
    def forward(self, q):
        seq_len = q.shape[1] # (seq_len), gets sequence length of query tensor q
        angles = self.angles[:seq_len] # gets angles up to a specific sequence length
        q_swapped = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q)
        final_output = (q * torch.cos(angles)) + (q_swapped * torch.sin(angles))
        return final_output

class Head(nn.Module):
    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.rope = RoPE(head_size, max_seq_len=block_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.rope(self.key(x)) # (B,T,C)
        q = self.rope(self.query(x)) # (B,T,C)
        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform weighted aggregation of values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# simple linear layer followed by non-linearity
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
      
    def forward(self, x):
        return self.net(x)

# Transformer block: communication followed by computation
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        # n_embd: embedding dimension, n_head: number of heads we would like
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
      
    def forward(self, x):
        # x + is for residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, dropout, vocab_size, block_size=128):
        super().__init__()
        # each token directly reads off the logits of the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
      
    def forward(self, idx, targets=None):
        B,T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # (B,T,C) aka (Batch, Time, Channel)
        # pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_embd #+ pos_embd # (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
      
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to before block size
            idx_cond = idx[:, -self.block_size:]
            # get the predictions, idx is (B,T) array of indices in current context
            logits, loss = self(idx_cond)
            # last time step
            logits = logits[:, -1, :] # (B,C)
            # probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # new characters
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx
