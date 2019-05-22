from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn import tree
import pydot
from sklearn import metrics
import pandas as pd
import numpy as np
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                     columns=iris['feature_names'] + ['target'])
# df['label'] = df.target.replace(dict(enumerate(df.target_names)))
print(df.head())
print(iris.feature_names)
print(iris.target_names)
print(df.describe())
x = iris['data']
y = iris['target']

iris_df = pd.DataFrame(x, columns=iris['feature_names'])
print(iris_df.head)
x, y = shuffle(x, y, random_state=0)  # random shuffle
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
classifier=DecisionTreeClassifier(criterion="entropy", max_depth=3) # To check accuracy
clf = classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
dot_data = StringIO()
tree.export_graphviz(classifier,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False,
                     proportion=True)
graph=pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris3.pdf")
