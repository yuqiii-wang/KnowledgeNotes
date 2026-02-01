import torch
import torch.nn.functional as F

d_model = 64

torch.manual_seed(42)

emb_A = F.normalize(torch.randn(1, d_model), p=2, dim=1)
emb_B = F.normalize(torch.randn(1, d_model), p=2, dim=1)
emb_C = F.normalize(torch.randn(1, d_model), p=2, dim=1)
emb_D = F.normalize(torch.randn(1, d_model), p=2, dim=1)

token_map = {'A': emb_A, 'B': emb_B, 'C': emb_C, 'D': emb_D}
history_seq = ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'A', 'B']
history_embs = torch.cat([token_map[t] for t in history_seq], dim=0)
print(f"Sequence History: [ {', '.join(history_seq)} ]")

K = history_embs

V = torch.roll(history_embs, shifts=-1, dims=0) 

user_input = input("Enter current token (A, B, C, or D): ").strip().upper()
if user_input == 'A':
    current_token = emb_A
elif user_input == 'B':
    current_token = emb_B
elif user_input == 'C':
    current_token = emb_C
elif user_input == 'D':
    current_token = emb_D
else:
    print("Invalid input, defaulting to A")
    user_input = 'A'
    current_token = emb_A

Q = current_token # [1, 64]

scores = torch.matmul(Q, K.T) # [1, seq_len]

attn_weights = F.softmax(scores * 10, dim=1) # *10 to sharpen (simulating trained head)

output = torch.matmul(attn_weights, V)

# --- 7. CHECK PREDICTIONS (LOGITS) ---
# Project output against vocab (A, B, C)
logit_A = torch.matmul(output, emb_A.T).item()
logit_B = torch.matmul(output, emb_B.T).item()
logit_C = torch.matmul(output, emb_C.T).item()
logit_D = torch.matmul(output, emb_D.T).item()

# Calculate Probabilities
probs = F.softmax(torch.tensor([logit_A, logit_B, logit_C, logit_D]), dim=0)

# --- 8. RESULTS ---
print(f"A logit: {logit_A:.4f}; probability: {probs[0]:.4f}")
print(f"B logit: {logit_B:.4f}; probability: {probs[1]:.4f}")
print(f"C logit: {logit_C:.4f}; probability: {probs[2]:.4f}")
print(f"D logit: {logit_D:.4f}; probability: {probs[3]:.4f}")