# PyTorch

* `unsqueeze`

```python
x = torch.tensor([1, 2, 3, 4])

torch.unsqueeze(x, 0)
# print:
# tensor([[ 1,  2,  3,  4]])

torch.unsqueeze(x, 1)
# print:
# tensor([[ 1],
#         [ 2],
#         [ 3],
#         [ 4]])
```

* `embedding`

Used to store word embeddings and retrieve them using indices. 

```python
embedding = nn.Embedding(30522, 768)
embedding([2013]) # get the 2013 token_id's embeddings
```