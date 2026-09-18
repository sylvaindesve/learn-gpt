# Dérivées partielles de la fonction de perte MSE en régression linéaire

## Le modèle

Pour une régression linéaire simple, la prédiction pour un exemple $i$ s'écrit :

$$\hat{y}_i = w x_i + b$$

- $x_i$ : variable d'entrée (feature) de l'exemple $i$
- $y_i$ : valeur réelle observée
- $\hat{y}_i$ : valeur prédite
- $w$ : poids (pente)
- $b$ : biais (ordonnée à l'origine)
- $n$ : nombre d'exemples

## La fonction de perte (MSE)

L'erreur quadratique moyenne (Mean Squared Error) s'écrit :

$$L(w, b) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - (w x_i + b) \right)^2$$

> **Remarque :** on rencontre souvent la variante $\frac{1}{2n}\sum(\dots)^2$. Le facteur $\frac{1}{2}$ n'a aucune signification statistique : il sert uniquement à simplifier le $2$ produit par la dérivation. Les deux conventions mènent au même minimum (seule l'échelle du gradient change, ce qui s'absorbe dans le taux d'apprentissage).

## Dérivée par rapport à $w$

### Poser le résidu

On note le résidu de l'exemple $i$ :

$$e_i = y_i - (w x_i + b)$$

La perte devient :

$$L = \frac{1}{n} \sum_{i=1}^{n} e_i^2$$

### Linéarité de la dérivation

La dérivée d'une somme est la somme des dérivées, et $\frac{1}{n}$ est une constante donc :

$$\frac{\partial L}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial}{\partial w} \left( e_i^2 \right)$$

### Théorème de dérivation des fonctions composées

On applique le [Théorème de dérivation des fonctions composées](https://fr.wikipedia.org/wiki/Th%C3%A9or%C3%A8me_de_d%C3%A9rivation_des_fonctions_compos%C3%A9es) (*Chain rule* en anglais). Ce théorème nous dit que $(g \circ f)'(a) = g'(f(a)) \cdot f'(a)$

Ainsi, pour $u \mapsto u^2$ appliquée à $e_i$ :

$$\frac{\partial}{\partial w} \left( e_i^2 \right) = 2 e_i \cdot \frac{\partial e_i}{\partial w}$$

### Dérivée interne

$$\frac{\partial e_i}{\partial w} = \frac{\partial}{\partial w} \left( y_i - w x_i - b \right) = 0 - x_i - 0 = -x_i$$

($y_i$ et $b$ sont des constantes vis-à-vis de $w$.)

### Assemblage

$$\frac{\partial L}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} 2 e_i \cdot (-x_i) = -\frac{2}{n} \sum_{i=1}^{n} x_i \left( y_i - (w x_i + b) \right)$$

**Forme équivalente (souvent utilisée en code) :**

$$\boxed{\ \frac{\partial L}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} x_i \left( \hat{y}_i - y_i \right)\ }$$

## Dérivée par rapport à $b$

### Linéarité de la dérivation

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial}{\partial b} \left( e_i^2 \right)$$

### Théorème de dérivation des fonctions composées

$$\frac{\partial}{\partial b} \left( e_i^2 \right) = 2 e_i \cdot \frac{\partial e_i}{\partial b}$$

### Dérivée interne

$$\frac{\partial e_i}{\partial b} = \frac{\partial}{\partial b} \left( y_i - w x_i - b \right) = -1$$

### Assemblage

$$\frac{\partial L}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} \left( y_i - (w x_i + b) \right)$$

**Forme équivalente :**

$$\boxed{\ \frac{\partial L}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} \left( \hat{y}_i - y_i \right)\ }$$

## Interprétation

| Gradient | Lecture intuitive |
|---|---|
| $\partial L / \partial w$ | Moyenne des erreurs **pondérée par $x_i$** : un exemple avec un grand $x$ pèse davantage sur la pente. |
| $\partial L / \partial b$ | Moyenne simple des erreurs : si le modèle sous-estime globalement, le gradient est négatif et $b$ augmente. |

Si les résidus sont centrés et décorrélés de $x$, les deux gradients s'annulent : c'est exactement la condition d'optimalité des moindres carrés.

---

## Mise à jour par descente de gradient

À chaque itération, avec un taux d'apprentissage $\alpha$ :

$$w \leftarrow w - \alpha \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

Le signe moins fait descendre la perte : on se déplace dans la direction opposée au gradient.
