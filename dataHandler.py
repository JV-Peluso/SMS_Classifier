import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
    
""" Class that load files, check it, and work with it """

rndSt = 16
printI = True
textF = ''
targF = ''
trainX = pd.DataFrame()
testX = pd.DataFrame()
trainY = pd.DataFrame()
testY = pd.DataFrame()
predFile = 'myData/SPAM_Pred.xlsx'

def loadCsv(fileName, encod = 'utf-8', skRows = 0):
    df = pd.read_csv(fileName, encoding = encod, skiprows = skRows)
    return df

def renameCols(df, cols):
    df.columns = cols
    return df

def checkAll(df, cols, newColName, oldT, newT, DelD = False, DelN = False, 
             dropC = False):
    
    ''' Check data integrity '''
    
    try:
        if isinstance(df, pd.DataFrame) : DF = df
    except:
        print('El archivo recibido no es un Pandas dataframe')

    printShape(DF)
    if printI : printInfo(DF)
    checkDup(DF, DelD)
    checkNaN(DF, DelN)
    mergeCorp(DF, cols, newColName, oldT, newT, dropC = dropC)
    print('Proceso completado!')
    return DF

def printShape(df):
    print('Estructura del Dataframe\n------------------------')
    print('{}\n'.format(df.shape))

def printInfo(df):
    print('Información del Dataframe\n-------------------------')
    print('{}\n'.format(df.info()))
    
def checkNaN(df, Del = False):
    print('Valores NaN\n-----------')
    for col in df.columns:
        nan = np.sum(df[col].isna().sum())
        print('{} --> {} Valores NaN'.format(col, nan))
        if nan > 0 and Del:
            df.dropna()
            print('{} líneas con valores NaN eliminadas correctamente\n'.format(nan))

def checkDup(df, Del = False):
    dup = df.duplicated().sum()
    print('Líneas duplicadas\n-----------------')
    if dup > 0 and Del:
        df.drop_duplicates(inplace = True)
        print('{} líneas duplicadas\nLíneas eliminadas correctamente\n'.format(dup))
    else : print('{} líneas duplicadas\n'.format(dup))
        
def mergeCorp(df, cols, newColName, oldT, newT, fillVal = '', dropC = False,
              dropT = True):
    
    ''' Process that merge SMS columns in one corpus, and creates new target feature '''
    
    print('\nActualizando valores NaN...')
    df.fillna(fillVal, inplace=True)
    print('Combinando cuerpos del SMS...')
    swich = True
    for col in cols:
        if swich :
            df[newColName] = df[col]
            swich = False
        else:
            df[newColName] = df[newColName] + df[col]
    if dropC:
        df.drop(cols, axis = 1, inplace = True) 
    print('Creando nueva etiqueta objetivo...\n')
    df[newT] = df[oldT].apply(lambda x : 1 if x == 'spam' else 0)
    if dropT:
        df.drop(oldT, axis = 1, inplace = True)     
    textF = newColName
    targF = newT

def groupDF(df, col, method = 'mean'):
    
    ''' Returns the DF statistical info selected '''
    
    summaryDF = df.groupby(col)
    if method == 'mean' : return summaryDF.mean()
    if method == 'std' : return summaryDF.std()
    
def outliersC(df, limit = 1.5):
    
    ''' Process that check for outliers '''
    
    lstResults = []
    lstColumns = ['Feature', 'Lower outliers', 'Upper outliers', 'Lower outliers pct', 
                  'Upper outliers pct']
    tmpDF = df.describe()
    for feature in list(tmpDF.columns):
        IQR = [tmpDF.loc['25%', feature], tmpDF.loc['75%', feature]]
        lowCount = df[df[feature] < ((IQR[0] - (IQR[1]-IQR[0])*limit))].count()[0]
        highCount = df[df[feature] > ((IQR[1] + (IQR[1]-IQR[0])*limit))].count()[0]
        lowPerc = (lowCount/df.shape[0])*100
        highPerc = (highCount/df.shape[0])*100
        lstResults.append([feature, lowCount, highCount, lowPerc, highPerc])
    df = pd.DataFrame(lstResults, columns = lstColumns).set_index('Feature')
    return df

def removeOutL(df, lstCols, limit = 2.0):
    
    ''' Process that remove outliers only for EDA '''
    
    tmpDF = df.describe()
    for feature in lstCols:
        IQR = [tmpDF.loc['25%', feature], tmpDF.loc['75%', feature]]
        outL = pd.DataFrame(df[df[feature] > ((IQR[1] + (IQR[1]-IQR[0])*limit))])
        dropRows(df, outL.index)
    return df

def dropRows(df, indexList):
    df.drop(indexList, inplace = True)
    df.reset_index(inplace = True, drop = True)

def bowCount(X, targV):
    
    ''' Top-10 words BoW '''
    
    textF = 'SMS'
    targF = 'SPAM'
    vect = CountVectorizer()
    bowMatrix = vect.fit_transform(X[X[targF] == targV][textF])
    bowDF = pd.DataFrame(bowMatrix.toarray())
    bowDF.columns = vect.get_feature_names()
    bowDF = bowDF.sum().reset_index()
    bowDF.rename(columns = {0:'Count', 'index' : 'Word'}, inplace = True)
    bowDF.sort_values(by='Count', ascending = False, inplace = True)
    return bowDF.head(10)

def dataPrep(df, feat, targ, TTsplit, strat, norm = False):
    
    ''' Preprocess data for modeling '''
    
    trainX, testX, trainY, testY = splitTrainTest(df, df[feat], df[targ], TTsplit, df[targ])
    if norm:
        trainX, testX = normalize(trainX, testX, feat)
    
    return trainX, testX, trainY, testY

def splitTrainTest(df, X ,y ,size, strat):
    
    ''' Split data in train and test arrays '''
    
    trX,trY,tsX,tsY = train_test_split(X, y, test_size=size, random_state=rndSt, stratify=strat)
    print('Datos separados en sets de aprendizaje y prueba')
    print('===============================================')
    print('Tamaño de los datos completos    --> ', df.shape)
    print('Propiedades de entrenamiento     --> ', trX.shape)
    print('Etiquetas de entrenamiento       --> ', tsX.shape)
    print('Propiedades de prueba            --> ', trY.shape)
    print('Etiquetas de prueba              --> ', tsY.shape,)
    print('===============================================\n')
    return trX,trY,tsX,tsY

def normalize(trDF, tsDF, cols):
    
    ''' Normalize (StandardScaler) data '''
    
    scaler = StandardScaler()
    trDF_N = pd.DataFrame(scaler.fit_transform(trDF), columns = cols, 
                          index = trDF.index)
    tsDF_N = pd.DataFrame(scaler.transform(tsDF), columns = cols, 
                          index =tsDF.index)
    print('Datos normalizados')
    return (trDF_N, tsDF_N)