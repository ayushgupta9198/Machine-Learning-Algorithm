def Multinomial123():
    from sklearn.datasets import load_iris
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix
    iris=load_iris() # call the class
    # print(iris)
    x=iris['data']
    y=iris['target']
    # print(x)
    # print(y)
    x,y=shuffle(x,y,random_state=0) # random shuffle
    # print(x)
    # print(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.40,random_state=0)
    # print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    classifier=GaussianNB();
    classifier.fit(x_train,y_train)
    y_prediction=classifier.predict(x_test)
    cm=confusion_matrix(y_test,y_prediction)
    # print(cm)
    # print("Multinomial Naive Model accuracy(in %):",metrics.accuracy_score(y_test,y_prediction)*100)
    abc1=metrics.accuracy_score(y_test,y_prediction)*100
    return(abc1)

b=Multinomial123()