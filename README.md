# Trabalho de classificação utilizando autosklearn AutoML
## Publicação deste trabalho 
[ARTIGO](http://bib.pucminas.br:8080/pergamumweb/vinculos/000071/00007123.pdf)

[VIDEO](https://www.youtube.com/watch?v=S991l1eaiCw)


# auto-sklearn

auto-sklearn é um kit de ferramentas de aprendizado de máquina automatizado e um substituto imediato para um estimador scikit-learn.

Encontre a documentação [here](http://automl.github.io/auto-sklearn/)

# Automated Machine Learning em poucas linhas
## Instalação
```python
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn import svm
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
```

## Execução
```python
# configure auto-sklearn
automl = autosklearn.classification.AutoSklearnClassifier(
          time_left_for_this_task=3600, # execute o auto-sklearn por no máximo x segundos
          per_run_time_limit=600, #, gastar no máximo Y segundos para cada modelo de treinamento
          )
# train model(s)
automl.fit(x_train, y_train)

# evaluate
y_hat = automl.predict(x_test)
```
## Métricas 

```python
test_acc = sklearn.metrics.accuracy_score(y_test, y_hat)
test_report = sklearn.metrics.classification_report(y_test, y_hat)
test_matrix = sklearn.metrics.confusion_matrix(y_test, y_hat)
print("-------------------------------------------------------------------------")
print("Test Accuracy score {0}".format(test_acc))
print("-------------------------------------------------------------------------")
print("Test Report score {0}".format(test_report))
print("-------------------------------------------------------------------------")
print("Test Confusion Matrix {0}".format(test_matrix))
print("-------------------------------------------------------------------------")
print(automl.sprint_statistics())
print("-------------------------------------------------------------------------")
print(automl.show_models())
print("-------------------------------------------------------------------------")
print(automl.get_models_with_weights())
```


## Publicações relevantes

Efficient and Robust Automated Machine Learning  
Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Springenberg, Manuel Blum and Frank Hutter  
Advances in Neural Information Processing Systems 28 (2015)  
http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf

Auto-Sklearn 2.0: The Next Generation  
Authors: Matthias Feurer, Katharina Eggensperger, Stefan Falkner, Marius Lindauer and Frank Hutter  
arXiv:2007.04074 [cs.LG], 2020
https://arxiv.org/abs/2007.04074
