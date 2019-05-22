def Gaussion123():
    from sklearn.datasets import load_iris # importing datasets
    from sklearn.utils import shuffle # to shuffle the datasets
    from sklearn.model_selection import train_test_split # to split the datasets
    from sklearn.naive_bayes import GaussianNB
    from sklearn import metrics # to check the accuracy
    iris=load_iris() # call the class
    print(iris)
    x=iris['data']
    y=iris['target']
    print(x)
    print(y)
    x,y=shuffle(x,y,random_state=0) # random shuffle
    print(x)
    print(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    model=GaussianNB() #call the class
    model.fit(x_train,y_train)
    predicted=model.predict(x_test)
    print(predicted)
    print(y_test)
    print("Gaussian Naive Bayes Model Accuracy(in %):",metrics.accuracy_score(y_test,predicted)*100)
    abc4=metrics.accuracy_score(y_test,predicted)*100
    return(abc4)
e=Gaussion123()