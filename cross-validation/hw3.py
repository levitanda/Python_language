from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt
import itertools
from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


#a
def cross_validation_error(X,y,model,folds):
    val_err = 0
    train_err=0
    sp=KFold(n_splits=folds)
    for train_i, test_i in sp.split(X):
        X_train, X_val=X[train_i], X[test_i]
        y_train, y_val=y[train_i], y[test_i]
        model=model.fit(X_train,y_train)
        X_train_pred=model.predict(X_train)
        X_val_pred=model.predict(X_val)
        val_err = val_err + 1 - accuracy_score(X_val_pred, y_val)
        train_err=train_err+1-accuracy_score(X_train_pred,y_train)
    return [train_err/folds, val_err/folds]

def SVM_results(X_train, y_train, X_test, y_test):
    model=svm.SVC(kernel='linear')
    linear_val=cross_validation_error(X_train, y_train, model, 5)
    fit_model=model.fit(X_train, y_train)
    test_pred=fit_model.predict(X_test)
    res_dict={}
    res_dict['svm_linear']=[linear_val[0],linear_val[1],1-accuracy_score(test_pred , y_test)]
    for d in [2,4,6,8,10]:
        modp=svm.SVC(kernel = 'poly', degree = d)
        poly_val=cross_validation_error(X_train, y_train, modp, 5)
        fit_model=modp.fit(X_train,y_train)
        test_pred=fit_model.predict(X_test)
        res_dict['svm_poly_degree_' + str(d)]=[poly_val[0],poly_val[1],1-accuracy_score(test_pred, y_test)]
    for g in [0.001,0.01,0.1,1,10]:
        modrbf=svm.SVC(kernel = 'rbf', gamma = g)
        rbf_val=cross_validation_error(X_train, y_train, model, 5)
        fit_model=modrbf.fit(X_train,y_train)
        test_pred=fit_model.predict(X_test)
        res_dict['svm_rbf_gamma_' + str(g)]=[rbf_val[0],rbf_val[1],1-accuracy_score(test_pred, y_test)]
    return res_dict

#b
def load_mnist():
    np.random.seed(2)
    (X, y), (_, _) = mnist.load_data()
    indexes = np.random.choice(len(X), 8000, replace=False)
    X = X[indexes]
    y = y[indexes]
    X = X.reshape(len(X), -1)
    return X, y
x , y = load_mnist()
X_train,X_test,y_train,y_test=train_test_split(x, y, test_size=0.25, random_state=98)

#c
scale=MinMaxScaler(feature_range=(-1, 1))
scale.fit(X_train)
X_train=scale.transform(X_train)
X_test=scale.transform(X_test)

#d
modelk=SVC(kernel='linear')
modelk.fit(X_train, y_train)
y_pred=modelk.predict(X_test)
cmap=plt.cm.Blues
cm=confusion_matrix(y_test, y_pred)
cm=cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title('SVM-kernel')
plt.colorbar()
f='.2f'
th=cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], f),
        horizontalalignment="center",
        color="white" if cm[i, j] > th else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#e
res_dict=SVM_results(X_train, y_train, X_test, y_test)
train_er=[]
val_er=[]
test_er=[]
for mod in res_dict.values():
    train_er.append(round(mod[0],2))
    val_er.append(round(mod[1],2))
    test_er.append(round(mod[2],2))
x=np.arange(len(train_er))
width=0.25
fig,ax=plt.subplots()
rec1=ax.bar(x - width, train_er, width, label='Train Error')
rec2=ax.bar(x, val_er, width, label='Validation Error')
rec3=ax.bar(x + width, test_er, width, label='Test Error')
ax.set_ylabel('Errors')
ax.set_title('Errors by different models')
ax.set_xticks(x)
ax.set_xticklabels(tuple(res_dict.keys()), fontsize=5)
ax.legend()
for rec in rec1:
    height=rec.get_height()
    plt.annotate('{}'.format(height), xy=(rec.get_x(), height), va='bottom')
for rec in rec2:
    height=rec.get_height()
    plt.annotate('{}'.format(height), xy=(rec.get_x(), height), va='bottom')
for rec in rec3:
    height=rec.get_height()
    plt.annotate('{}'.format(height), xy=(rec.get_x(), height), va='bottom')
fig.tight_layout()
plt.show()
