import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

''' Class that plot graphs '''
    
def setStyle(style = 'white'):
    sns.set(style = style)

def plotBarText(df, fig, title, xLabel, xTicks, xLabelFS = 10, xTicksFS = 12,
                plotC = 'b', barW = 0.65, barTxtFS = 10, titleFS = 10):
    
    ''' Process that draw a tuned bar plot with text above the bars '''
    
    _ = plt.bar(xTicks, df.iloc[:, 0], width = barW, color = plotC, alpha = 0.75)
    plt.title(title, fontsize = titleFS, fontweight = 450)
    plt.xlabel(xLabel, fontsize = xLabelFS)
    plt.tick_params(top=False,bottom=False,left=False,right=False,labelleft=False,labelbottom=True)
    plt.xticks(fontsize = xTicksFS)
    plt.grid(b=False)
    b,t = plt.ylim()
    plt.ylim(top=(t*1.075))
    for spine in plt.gca().spines.values():
        spine.set_visible(False) if spine.spine_type !='bottom' else spine.set_visible(True)
    for bar in _:
        height = bar.get_height()
        txtHeight = str(np.around(height,decimals=2))+'%'

        plt.gca().text(bar.get_x() + bar.get_width()/1.85, (bar.get_height()+0.85), txtHeight,
                       ha='center', color='black', fontsize=barTxtFS)

def plotKDE(fig, title, kde, featureD, labels, 
            colors = ['b', 'g'], legLoc = 0, legFS = 12, titleFS = 15):
    
    ''' Process that draw KDE plots '''
    
    ax=sns.kdeplot(kde[0], color=colors[0], shade=True, label= labels[0])
    ax=sns.kdeplot(kde[1], color=colors[1], shade=True, label= labels[1])
    plt.title('{} {}'.format(featureD, title), fontsize = titleFS)
    plt.tick_params(top=False, bottom=True, left=True, right=False, labelsize = 8)
    plt.xlabel('',fontsize=16);
    plt.ylabel('Density',fontsize=8);
    if legLoc == 0 :
        plt.legend(frameon=False, loc = legLoc, fontsize = legFS)
    else :    
        plt.legend(frameon=False, bbox_to_anchor=(legLoc), fontsize = legFS)
    plt.tight_layout()

def plotCorr(corr, title, figSize = [5,5], titleFS = 15, cmap = 'blues', annot = True, 
            square = True, fmt = '.2f', vMM = [-1,1], lineW = 0.25, cbarD = {}, rot = 90,
            annD = {}, ticksFS = 8, yLim = [0.0,0.0]):
    
    ''' Process that plot a correlation matrix '''

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(figSize))
    ax.set_title(title, fontdict={'fontsize': titleFS})
    sns.heatmap(corr, annot = annot, mask=mask, cmap=cmap, vmin=vMM[0], vmax=vMM[1],
                center=0, fmt=fmt, annot_kws=annD, square=True, linewidths=lineW, 
                cbar_kws=cbarD)
    plt.xticks(fontsize = ticksFS, rotation = rot, ha = 'right')
    plt.yticks(fontsize = ticksFS)
    plt.title(title, fontsize = titleFS)
    b,t = plt.ylim()
    plt.ylim(bottom=(b*yLim[0]), top = (t*yLim[1]))
    
def plotBox(df, fig):
    ax = sns.boxplot(df, orient = 'h', palette = "muted")
    
def plotBarH(X, y, fig, title, width, nType = '', symbol = '', 
             fontS = 10, plotC = 'b', sep = [1,2]):
    
    ''' Process that draw a tuned horizontal bar plot with text on the bars '''
    
    setStyle('white')
    br = plt.barh(X, y, color = plotC, alpha = 0.7)
    plt.title(title, fontsize = 11, y = 1.05)
    plt.yticks(fontsize = 9)
    plt.xticks(fontsize = 0)
    plt.rcParams['axes.facecolor'] = 'white'
    l, r = plt.xlim()
    plt.xlim(left=(y.min()*0.6))
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=True, labelbottom=False)
    
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    for bar in br:
        width = bar.get_width()
        sepN = sep[0] if width < 100 else sep[1]
        width  = str('{0:.0f}'.format(width)) + symbol
        plt.gca().text((bar.get_width()-sepN), bar.get_y() + bar.get_height()/3.25, str(width),
                       ha='center', color='black', fontsize=fontS)
        
def plotConfMatrix(clfN, yTest, yPred, target_names, title='Confusion matrix', 
                   cmap=None, figSize = [8,6], normalize=True):
    
    ''' Process that draw the confussion matrix for given predictions '''
    
    cm = confusion_matrix(yTest, yPred)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    fig = plt.figure(figsize=(figSize))
    if cmap is None: cmap = plt.get_cmap('Blues')
    if normalize: cm = (cm.astype('float')*100) / cm.sum(axis=1)[:, np.newaxis]

    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=cmap)
    if normalize : fig.colorbar(cax, ticks=np.arange(0,101,20))
    else : fig.colorbar(cax)
    ax.set_xticklabels([''] + target_names)
    ax.set_yticklabels([''] + target_names)
    plt.tick_params(axis='both', which='major', length=0)
    plt.ylabel('True', fontsize = 12)
    plt.xlabel('Predicted', fontsize = 12)
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize: plt.text(j, i, "{:0.2f}%".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontweight = 568, fontsize = 12)
        else: plt.text(j, i, "{:,}".format(cm[i, j]),
                       horizontalalignment="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontweight = 568, fontsize = 12)
    plt.tight_layout()
    ax.xaxis.set_label_coords(0.5, -0.075)
    
    # Print scores
    ax.text(2.55,-0.45,'== '+clfN+' ==',fontsize=12)
    ax.text(2.55,-0.20,'Accuracy: '+str(round(accuracy_score(yTest, yPred),4)),fontsize=12)
    ax.text(2.55,0.0,'Precision: '+str(round(precision_score(yTest, yPred),4)),fontsize=12)
    ax.text(2.55,0.2,'Recall: '+str(round(recall_score(yTest, yPred),4)),fontsize=12)
    ax.text(2.55,0.4,'F1: '+str(round(f1_score(yTest, yPred),4)),fontsize=12)
    
def plotFeatImp(X, Y, fig, title, xLabel, xLabelFS = 10, rot = 45,
                xTicksFS = 11, plotC = 'b', barW = 0.9, barTxtFS = 11, simb = ''):
    
    ''' Process that draw bar plot for the feature importance '''
    
    _ = plt.bar(X, Y, width = barW, color = plotC)
    plt.title(title, fontsize = 15, fontweight = 450)
    plt.xlabel(xLabel, fontsize = xLabelFS)
    plt.tick_params(top=False,bottom=False,left=False,right=False,labelleft=False,labelbottom=True)
    plt.xticks(fontsize = xTicksFS, rotation = rot, ha = 'right')
    plt.grid(b=False)
    b,t = plt.ylim()
    plt.ylim(top=(t*1.075))
    for spine in plt.gca().spines.values():
        spine.set_visible(False) if spine.spine_type !='bottom' else spine.set_visible(True)
    i = 0
    for bar in _:
        if str(X.iloc[i])[:5] == 'basic':
            bar.set_color('g')
        elif str(X.iloc[i])[:5] == 'tfidf':
            bar.set_color('b')
        else:
            bar.set_color('r')  
        i = i +1
        height = bar.get_height()
        txtHeight = str(np.around(height,decimals=2))+simb
        plt.gca().text(bar.get_x() + bar.get_width()/1.85, (bar.get_height()+0.01), txtHeight,
                       ha='center', color='black', fontsize=barTxtFS)
    colors = {'Basic':'g', 'TF-IDF':'b', 'CountV' : 'r'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)