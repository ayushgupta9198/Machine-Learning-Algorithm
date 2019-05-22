def Bernauli123():
    from sklearn.datasets import load_iris
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import BernoulliNB
    from sklearn import metrics
    iris=load_iris() # call the class
    # print(iris)
    x=iris['data']
    y=iris['target']
    # print(x)
    # print(y)
    x,y=shuffle(x,y,random_state=0) # random shuffle
    # print(x)
    # print(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)
    # print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    model=BernoulliNB();
    model.fit(x_train,y_train)
    predicted=model.predict(x_test)
    # print(predicted)
    # print(y_test)
    # print("Bernauli Model Accuracy(in %):",metrics.accuracy_score(y_test,predicted)*100)
    abc3=metrics.accuracy_score(y_test,predicted)*100
    return(abc3)
d=Bernauli123()