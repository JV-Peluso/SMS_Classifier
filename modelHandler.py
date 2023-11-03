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
        print('      {} {} --> {:.4f}'.format(scoring.capitalize(), spc, score))
    print('================================')

def paramTuning(X, y, pipe, pipeN, parGrid, cv = 5, score = 'accuracy', vrb = 0):
    
    ''' Hyperparameter tuning process '''
    
    cols = ['params']
    for sc in score:
        cols.append('mean_test_' + sc)
        
    gridSCV = GridSearchCV(estimator = pipe, param_grid = parGrid, cv= cv,
                           scoring = score, n_jobs = -1, verbose = vrb, refit= False)
    gridSCV.fit(X, y)
    gscvDF = pd.DataFrame(gridSCV.cv_results_)
    rstDF = gscvDF[gscvDF['rank_test_' + score[-1]] == 1][cols].iloc[0]
    strHead = '='*(len(pipeN)+6)
    print('{}\n== {} ==\n{}\n'.format(strHead, pipeN, strHead))
    print('Parametros: {}\n'.format(rstDF[cols[0]]))
    for i in range(len(score)):
        spc = " "*(9-len(score[i]))
        print('{} {} --> {:.4f}'.format(score[i].capitalize(), spc, rstDF[cols[i+1]]))

def modelValidation(lstPipe, lstPipeN, lstLbl, figSizeCM = [6,3]):
    
    ''' Process that validates models and print results '''

    for i in range(len(lstPipe)):
        lstPipe[i].fit(data.trainX, data.trainY)
        pred = lstPipe[i].predict(data.testX)
        pg.setStyle(style = 'white')
        pg.plotConfMatrix(lstPipeN[i], data.testY, pred, lstLbl, 
                          cmap='GnBu', figSize = figSizeCM, normalize = True)
        plt.show()

def bestModel(bestM, fig, top = 10):
    
    ''' Process that predicts and print feature importance with the final classifier '''
    
    print('{}\n== BEST MODEL ==\n{}\n'.format('='*16, '='*16))
    print('     {}\n'.format(bestM.named_steps['clf']))
    print('Fitting model to train data...')
    bestM.fit(data.trainX, data.trainY)
    self.featImp(bestM, fig, top = top)
    print('\nPredicting with test data...')
    pred = bestM.predict(data.testX)
    savePred(pred)

def featImp(model, fig, top):
    
    ''' Plot feature importance of the best model '''
    
    imp = model.named_steps['clf'].coef_[0].tolist()
    names = model.named_steps['feat'].get_feature_names()
    featDF = pd.DataFrame(list(zip(names, imp)), 
                          columns =['Name', 'Imp']) 
    featDF.sort_values(by='Imp', ascending = False, inplace = True)
    featDF = featDF.head(top)
    pg.setStyle('white')    
    pg.plotFeatImp(featDF['Name'], featDF['Imp'], fig, 'Feature importance', '')
    plt.show()

def savePred(pred):
    
    ''' Save predictions to an excel file '''
    
    finalDF = pd.DataFrame(pred)
    finalDF.to_excel(data.predFile, index=False)
    print('Predictions exported') 