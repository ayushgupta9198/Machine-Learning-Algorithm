import matplotlib.pyplot as plt
from matplotlib import pyparsing
import numpy as np
from Python_class.Decisiontree import a
from Python_class.SVM import c
from Python_class.Multinomialnaivebayes import b
from Python_class.Bernauliclassifier import d
from Python_class.Gaussionalgorithm import e
from Python_class.Kneighbour import f
x1=[a,b,c,d,e,f]
print(x1)
abc1=['Decisiontree','Mutlinominal','Svm','Bernauli','Gaussion','Knearest']
plt.bar(abc1,x1,align="center")
plt.yticks(np.arange(0,100,5))
plt.show()
