import torch
import torch.nn.functional as F
from transformer import GPT


with open('ye_lyrics.txt', 'r', encoding='utf-8') as f:
    text = f.read()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
chars = sorted(list(set(text)))
vocab_size = len(chars)

m = GPT()
m.load_state_dict(torch.load('./kanye-models/iter-final.pt'))
m = m.to(device=device)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# for _ in range(max_new):
block_size = 256
idx = torch.zeros((1, 1),dtype=torch.long, device=device)
while True:
    idx_cond = idx[:, -block_size:]
    logits, loss = m(idx_cond)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    print(decode(idx_next[0].tolist()), end="")
    idx = torch.cat((idx, idx_next), dim=1)
