def vectormachine():
    from sklearn.datasets import load_iris # importing datasets
    from sklearn.utils import shuffle # to shuffle the datasets
    from sklearn.model_selection import train_test_split # to split the datasets
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report,confusion_matrix
    from sklearn import metrics # to check the accuracy
    iris=load_iris() # call the class
    # print(iris)
    x=iris['data']
    y=iris['target']
    # print(x)
    # print(y)
    x,y=shuffle(x,y,random_state=0) # random shuffle
    # print(x)
    # print(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.50)
    # print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    svcclassifier=SVC(kernel='linear')
    svcclassifier.fit(x_train,y_train)
    y_prediction=svcclassifier.predict(x_test)
    # print(confusion_matrix(y_test,y_prediction))
    # print(classification_report(y_test,y_prediction))
    # print('SVM Model accuracy (in %):',metrics.accuracy_score(y_test,y_prediction)*100)
    abc2=metrics.accuracy_score(y_test,y_prediction)*100
    return(abc2)

c=vectormachine()