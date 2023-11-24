import torch
import torch.nn as nn
from torch.nn import functional as F
from time import time
import tiktoken

enc = tiktoken.get_encoding("p50k_base")

batch_size = 32
block_size = 128
n_embd = 32
# max_iters = 1
save_interval = 2000
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
eval_iters= 100
# n_head = 6
# n_layer = 6
n_head = 2
n_layer = 2
droupout = 0.2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with open('ye_lyrics.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# chars = sorted(list(set(text)))
# vocab_size = len(chars)
vocab_size = enc.n_vocab

# stoi = { ch:i for i, ch in enumerate(chars) }
# itos = { i:ch for i, ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[i] for i in l])
def encode(t): return enc.encode(t)
def decode(t): return enc.decode(t)
torch.manual_seed(1337)

data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data= train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))

        self.droupout = nn.Dropout(droupout)
    
    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.droupout(wei)

        v = self.value(x)
        out = wei @ v
        print(out.shape)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(droupout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attention = nn.Linear(n_embd, n_embd * 3)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(droupout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    
    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attention(x).split(n_embd, dim=2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

        wei = (q @ k.transpose(-1, -2)) * C ** -0.5
        wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(droupout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # self.sa = MultiHeadAttention(n_head, head_size)
        self.sa = CausalAttention()
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        # self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new):
        for _ in range(max_new):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    model = GPT()
    print(f"n params: {sum(p.numel() for p in model.parameters())}")
    print('loading model...')
    model.load_state_dict(torch.load('./kanye-models/iter-final.pt'))
    print('model loaded!')
    m = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters + 1):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if iter % save_interval == 0:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print(decode(m.generate(context, max_new=1000)[0].tolist()))
            torch.save(m.state_dict(), f"./kanye-models/iter-{iter}.pt")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new=1000)[0].tolist()))
    torch.save(m.state_dict(), f"./kanye-models/iter-final.pt")
