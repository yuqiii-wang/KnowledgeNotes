import numpy as np



data = """
After graduation from university with a master degree in AI, Yuqi started his career as a SLAM engineer specializing in computer vision recognition.
After one year, he accepted another job offer as a c++ engineer.
Since them, he has not changed his job.
"""

#####
#   Regard each char as a token.
#   Train an rnn to learn to predict the right sequence of chars forming the above `data`
#####

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size)) 
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }


##### hyperparameters
hidden_size = 768 # size of hidden layer of neurons
seq_length = 30 # number of steps to unroll the RNN for
learning_rate = 1e-1

class RNN:
    
    
    def __init__(self):
        # model parameters
        self.Wh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
        self.Uh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
        self.Wz = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
        self.bh = np.zeros((hidden_size, 1)) # hidden bias
        self.bz = np.zeros((vocab_size, 1)) # output bias
        self.hprev = np.zeros((hidden_size,1)) # h_{-1} init for mempry when token_idx = 0
        
        # backward pass: compute gradients going backwards
        self.dWh, self.dUh, self.dbh = \
            np.zeros_like(self.Wh), np.zeros_like(self.Uh), np.zeros_like(self.bh)
        self.dWz, self.dbz = \
            np.zeros_like(self.Wz), np.zeros_like(self.by)
        self.dhnext = np.zeros_like(self.h[0])
        
        # temp values for back-propogation
        self.h = np.zeros((len(x), hidden_size, 1))
        self.h[0] = np.copy(self.hprev)
        self.z = np.zeros((len(x), vocab_size, 1))
        self.p = np.zeros((len(x), vocab_size, 1))
        self.seq_len = 0
        self.loss = 0.0


    def forward(self, x):
        assert (len(x) < seq_length)
        self.seq_len = len(x)
        
        self.loss = 0.0
        pred_chars = ''
        
        ## xs: a token sequence that indicates at which pos the token is present and update: 0 -> 1
        xs = np.zeros((len(x), vocab_size, 1))
        for t_idx in range(0, len(x)):
            
            xs[t_idx][x[t_idx]] = 1 # set the used token to one
            ht_prev = self.hprev if t_idx - 1 < 0 else h[t_idx-1]
            
            zh = np.dot(self.Wh, xs[t_idx]) + np.dot(self.Uh, ht_prev) + bh
            self.h[t_idx] = np.tanh(zh)
            self.z[t_idx] = np.dot(self.Wz, h[t_idx]) + self.bz
            self.p[t_idx] = np.exp(z[t_idx]) / np.sum(np.exp(z[t_idx])) # probabilities for next chars
            self.loss += -np.log(p[t_idx][targets[t_idx],0])
            
            t_chosen_idx = np.argmax(p[t_idx])
            pred_chars += ix_to_char[t_chosen_idx]
    
        return pred_chars, loss

    def backward(self):
        for t_idx in reversed(range(0, self.seq_len )):
            
            ht_prev = self.hprev if t_idx - 1 < 0 else h[t_idx-1]


            dy = np.copy(p[t_idx])
            dy[targets[t]] -= 1
            

for epoch in range(100):
    
    p = 0
    while (p+seq_length+1 <= len(data)):
        
        ### Prepare inputs, the target is next char prediction
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
        
        
        
        p += 1
        
        

## rnn one layer init
x = np.zeros((len(inputs), vocab_size, token_emb_size))
Wh = np.random.rand(token_emb_size, token_emb_size)
Uh = np.random.rand(token_emb_size, token_emb_size)
h = np.random.rand(len(inputs),vocab_size, token_emb_size)
bh = np.zeros((vocab_size, token_emb_size))
Wz = np.random.rand(vocab_size, token_emb_size)
bz = np.zeros((vocab_size, 1))
z = np.zeros((len(inputs), vocab_size, 1))
y = np.zeros((len(inputs), vocab_size, 1))



# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

### rnn forward
loss = 0
for t in range(1, len(inputs)):
    h[t] = np.tanh(np.dot(x[t], Wh) + np.dot(h[t-1], Uh) + bh) # hidden state
    z[t] = np.dot(Wz, h[t]) + bz # unnormalized log probabilities for next chars
    y[t] = np.exp(z[t]) / np.sum(np.exp(z[t])) # probabilities for next chars
    # loss += -np.log(y[t][targets[t],0]) # softmax (cross-entropy loss)