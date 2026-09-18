# Les tenseurs de PyTorch

Un `torch.Tensor` est une matrice multidimensionnelle contenant des éléments du même type.

```python
x = torch.tensor([1.0, 2.0, 3.0])
```

La forme (i.e. les dimensions) d'un tenseur s'obtient avec `.shape` :

```python
x = torch.tensor([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
print(x.shape)  # torch.Size([2, 3])
```

La valeur d'un tenseur se récupère avec `.item()` :

```python
perte = torch.tensor(2.5)
print(perte)  # tensor(2.5000)
print(perte.item())  # 2.5
```

Il est possible de faire des opérations sur des tenseurs :

```python
x = torch.tensor([1, 2, 3])
w = torch.tensor(5)
b = torch.tensor(2)

print(w * x + b)  # tensor([ 7, 12, 17])
```

Dont le produit matriciel :

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[7, 8], [9, 10], [11, 12]])

print(a @ b)  # tensor([[ 58,  64], [139, 154]])
```

Mais surtout, les tenseurs supporte l'autograd, c'est-à-dire le calcul automatique du gradient en parcourant en sens inverse le graphe des opérations appliquées :

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])  # heures d'étude par semaine
y = torch.tensor([4.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0])  # note sur 20

w = torch.tensor(1.7, requires_grad=True)
b = torch.tensor(2.3, requires_grad=True)

y_pred = x * w + b
loss_mse = ((y_pred - y) ** 2).mean()

# Déclenche le calcul des gradients
loss_mse.backward()

print(w.grad)  # tensor(-5.3714)
print(b.grad)  # tensor(-1.3143)
```
