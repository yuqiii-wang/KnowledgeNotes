import numpy as np
import re

parameters = {}


def initialize_parameters(input_size, hidden_size, output_size):
    # parameters['Wxh'] = np.random.randn(hidden_size, input_size) * 1/np.sqrt(input_size)
    # parameters['Whh'] = np.random.randn(hidden_size, hidden_size) * 1/np.sqrt(input_size)
    # parameters['Why'] = np.random.randn(output_size, hidden_size) * 1/np.sqrt(input_size)
    parameters['Wxh'] = np.random.randn(hidden_size, input_size) * np.sqrt(2 / (hidden_size + input_size))
    parameters['Whh'] = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 /(hidden_size + hidden_size))
    parameters['Why'] = np.random.randn(output_size, hidden_size) * np.sqrt(2 /(output_size + hidden_size))
    # parameters['Wxh'] = np.random.rand(hidden_size, input_size) * np.sqrt(6) / np.sqrt(hidden_size + input_size)
    # parameters['Whh'] = np.random.rand(hidden_size, hidden_size) * np.sqrt(6) / np.sqrt(hidden_size + hidden_size)
    # parameters['Why'] = np.random.rand(output_size, hidden_size) * np.sqrt(6) / np.sqrt(output_size + hidden_size)
    # parameters['Wxh'] = np.random.rand(hidden_size, input_size) * 0.1
    # parameters['Whh'] = np.random.rand(hidden_size, hidden_size) * 0.1
    # parameters['Why'] = np.random.rand(output_size, hidden_size) * 0.1
    # parameters['Wxh'] = (np.random.rand(hidden_size, input_size)-0.5) * 0.1
    # parameters['Whh'] = (np.random.rand(hidden_size, hidden_size)-0.5) * 0.1
    # parameters['Why'] = (np.random.rand(output_size, hidden_size)-0.5) * 0.1
    parameters['bh'] = np.zeros((hidden_size, 1))
    parameters['by'] = np.zeros((output_size, 1))
    return parameters

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # for numerical stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
def rnn_forward(x, h_prev, parameters):
    Wxh, Whh, Why, bh, by = parameters['Wxh'], parameters['Whh'], parameters['Why'], parameters['bh'], parameters['by']
    h_next = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh)
    z = np.dot(Why, h_next) + by
    y = softmax(z)
    return y, h_next

def rnn_backward(x, h_prev, h_next, y_pred, y, parameters):
    Wxh, Whh, Why, bh, by = parameters['Wxh'], parameters['Whh'], parameters['Why'], parameters['bh'], parameters['by']
    dy = y_pred - y  # Gradient of loss with respect to output
    dWhy = np.dot(dy, h_next.T)
    dby = dy
    dh = np.dot(Why.T, dy) * (1 - h_next ** 2)
    dWxh = np.dot(dh, x.T)
    dWhh = np.dot(dh, h_prev.T)
    dbh = dh
    return dWxh, dWhh, dWhy, dbh, dby, dh

def update_parameters(parameters, grads, learning_rate):
    for param in parameters:
        parameters[param] -= learning_rate * grads[param]

def rnn_train(X, Y, input_size, hidden_size, output_size, epochs, learning_rate):
    parameters = initialize_parameters(input_size, hidden_size, output_size)
    early_termination_epsilon = 1e-05
    loss_history = []
    
    prev_epoch_total_loss = 0
    for epoch in range(epochs):
        h_prev = np.zeros((hidden_size, 1))
        total_loss = 0

        learning_rate = learning_rate * 0.999 ** epoch if epoch < 100 else \
                        learning_rate * 0.998 ** epoch if epoch < 200 \
                        else learning_rate
        
        for t in range(len(X)):
            x = X[t].reshape(-1, 1)
            y = Y[t].reshape(-1, 1)
            
            y_pred, h_next = rnn_forward(x, h_prev, parameters)
            loss = -np.sum(y * np.log(y_pred))  # Cross-entropy loss
            total_loss += loss
            
            dWxh, dWhh, dWhy, dbh, dby, dh = rnn_backward(x, h_prev, h_next, y_pred, y, parameters)
            
            grads = {
                'Wxh': dWxh,
                'Whh': dWhh,
                'Why': dWhy,
                'bh': dbh,
                'by': dby
            }
            
            update_parameters(parameters, grads, learning_rate)
            
            h_prev = h_next
        
        loss_history.append(total_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss}')

        if abs(prev_epoch_total_loss - total_loss) < early_termination_epsilon:
            break
        prev_epoch_total_loss = total_loss
    
    return parameters, loss_history

def do_one_hot_encoding(vocab: list):
    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {word_to_index[word]: word for word in word_to_index}
    # One-hot encoding
    one_hot_encoded = np.zeros((len(tokens), vocab_size), dtype=int)
    for i, token in enumerate(tokens):
        one_hot_encoded[i, word_to_index[token]] = 1
    return one_hot_encoded, word_to_index, index_to_word

if __name__ == "__main__":
    from train_data import train_seq_data
    tokens = re.findall(r'\b\w+\b|[^\s\w]', train_seq_data)
    tokens.append('.') # add an end token
    vocab = sorted(set(tokens))
    vocab_size = len(vocab)

    one_hot_encoded, word_to_index, index_to_word = do_one_hot_encoding(vocab)

    X = [ one_hot_encoded[word_to_index[token]] for token in tokens]
    Y = [ one_hot_encoded[word_to_index[tokens[idx+1]]] 
        if (idx+1) < len(tokens)-1 and token != '.' and token != '\n' 
            else one_hot_encoded[word_to_index.get('.')]
        for idx, token in enumerate(tokens)]
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Example usage
    input_size = vocab_size 
    hidden_size = 768  # Number of hidden units
    output_size = vocab_size 
    epochs = 300  # Number of epochs
    learning_rate = 0.01  # Learning rate

    parameters, loss_history = rnn_train(X, Y, input_size, hidden_size, output_size, epochs, learning_rate)

    y_pred_token_idx_ls = []
    h_prev = np.zeros((hidden_size, 1))
    for t in range(len(X)):
        x = X[t].reshape(-1, 1)
        y = Y[t].reshape(-1, 1)
        
        y_pred, h_next = rnn_forward(x, h_prev, parameters)
        h_prev = h_next
        y_pred_token_idx = np.argmax(y_pred)
        y_pred_token_idx_ls.append(y_pred_token_idx)

    y_pred_tokens = [index_to_word[idx] for idx in y_pred_token_idx_ls]
    print(" ".join(y_pred_tokens))