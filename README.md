# dtreeplt
it draws Decision Tree not using Graphviz, but only matplotlib.
## Output Image using proposed method (using only matplotlib)
![graphviz](output/result.png)
## Output Image using conventional method (Using Graphviz)
![graphviz](output/using_graphviz.png)

## Installation
`pip install dtreeplt`  
Requirements are numpy(>=1.15.1) and matplotlib(>=3.0.2) only.  
Python 3.6.X.

## Usage
### Quick Start
```python
from dtreeplt import dtreeplt
dtree = dtreeplt()
dtree.view()
```
### Using trained DecisionTreeClassifier
```python
# You should prepare trained model,feature_names, target_names.
# in this example, use iris datasets.
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from dtreeplt import dtreeplt

iris = load_iris()
model = DecisionTreeClassifier()
model.fit(iris.data, iris.target)

dtree = dtreeplt(
    model=model,
    feature_names=iris.feature_names,
    target_names=iris.target_names
)
fig = dtree.view()
#if you want save figure, use savefig method in returned figure object.
#fig.savefig('output.png')
```

