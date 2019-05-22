def K123():
    from sklearn.datasets import load_iris
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn import metrics
    iris=load_iris() # call the class
    print(iris)
    x=iris['data']
    y=iris['target']
    print(x)
    print(y)
    x,y=shuffle(x,y,random_state=0) # random shuffle
    print(x)
    print(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("K-Neighbour Model accuracy(in %):",metrics.accuracy_score(y_test,y_pred)*100)
    abc5=metrics.accuracy_score(y_test,y_pred)*100
    return(abc5)
f=K123()