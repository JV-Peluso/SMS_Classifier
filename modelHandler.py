import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion

''' Class that handles the ML models and its processes '''

def __init__(self):
    pass

def baseline(Xtrain, Xtest, clf, lstScoring, cv = 5):
    
    ''' Process that creates the baseline '''
    
    print('     Métricas de referencia')
    print('===============================')
    for scoring in lstScoring:
        spc = " "*(9-len(scoring))
        score = cross_val_score(clf, Xtrain, Xtest, cv=cv, scoring = scoring).mean()
        print('     {} {} --> {:.4f}'.format(scoring.capitalize(), spc, score))
    print('================================')

def modelValidation(trainX, trainY, testX, testY, lstPipe):
    
    ''' Process that validates models and print results '''
    
    lstPred = []
    for i in range(len(lstPipe)):
        lstPipe[i].fit(trainX, trainY)
        pred = lstPipe[i].predict(testX)
        #pg.setStyle(style = 'white')
        #pg.plotConfMatrix(lstPipeN[i], testY, pred, lstLbl, 
        #                 cmap='GnBu', figSize = figSizeCM, normalize = True)
        #plt.show()
        lstPred.append(pred)
        
    return lstPred

def bestModel(trainX, trainY, testX, bestM, top = 10):
    
    ''' Process that predicts and print feature importance with the final classifier '''
    
    print('{}\n== BEST MODEL ==\n{}\n'.format('='*16, '='*16))
    print('     {}\n'.format(bestM.named_steps['clf']))
    print('Fitting model to train data...')
    bestM.fit(trainX, trainY)
    print('\nPredicting with test data...')
    pred = bestM.predict(testX)
    savePred(pred)
    return bestM


def savePred(pred):
    
    ''' Save predictions to an excel file '''
    
    finalDF = pd.DataFrame(pred)
    finalDF.to_excel('myData/SPAM_Pred.xlsx', index=False)
    print('Predictions exported\n') 