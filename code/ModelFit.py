# -*- coding: utf-8 -*-
import FeatureExtraction as FE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,KFold
import logging
import numpy as np
import os 
import pandas as pd
from sklearn.base import BaseEstimator,ClassifierMixin,TransformerMixin,clone
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
import xgboost as xgb
import warnings 
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

cwd=os.getcwd()


def modelTest1():
    '''
    this is a rough test for train_
    '''
    clf=DecisionTreeClassifier(criterion='gini')
    train_path=os.path.join(os.path.dirname(cwd),'data/train_.csv')
    train=pd.read_csv(train_path,header=0,index_col=0)
    y=train.pop('Survived')
    X=train.values
    scores=cross_val_score(clf,X=X,y=y,cv=5)
    print(np.mean(scores))
    #we get a result of 0.778 not too good but we have a model

def modelTest2():
    train_path=os.path.join(os.path.dirname(cwd),'data/train_.csv')
    train=pd.read_csv(train_path,header=0,index_col=0)
    # add FamilySize to train
    train=FE.getFamilySize(train)
    features=['Age','Pclass','FamilySize','_Title_','_HasCabin_','_Embarked_']
    y=train.pop("Survived").values
    X=train[features].values
    rf=RandomForestClassifier(100)
    bg=GradientBoostingClassifier(n_estimators=100)
    xg=xgb.XGBClassifier(n_estimators=100)
    tree=DecisionTreeClassifier()
    stacking=StackingModel([rf,bg,xg],tree,cv=5)
    scores=cross_val_score(stacking,X=X,y=y,cv=5)
    print('modelTest2:',np.mean(scores))
 
# a stacking model for out of fold prediction 
class StackingModel(BaseEstimator,ClassifierMixin,TransformerMixin):
    def __init__(self,basemodels,metamodel,cv):
        self.basemodels=basemodels
        self.metamodel=metamodel
        self.cv=cv
    def fit(self,X,y):
        self.basemodels_=[list() for model in self.basemodels]
        out_of_fold=np.zeros((X.shape[0],len(self.basemodels)))
        kf=KFold(n_splits=self.cv,shuffle=True,random_state=86)
        for i,model in enumerate(self.basemodels):
            for train_index,test_index in kf.split(X,y):
                clone_=clone(model)
                clone_.fit(X[train_index],y[train_index])
                out=clone_.predict(X[test_index])
                out_of_fold[test_index,i]=out
                self.basemodels_[i].append(clone_)
        self.metamodel_=clone(self.metamodel)
        self.metamodel_.fit(out_of_fold,y)
        return self
    def __vote(self,votes_list):
        l=[]
        for votes in votes_list:
            v_=map(lambda x:1 if x>0 else -1,votes)
            vote=0
            for i in v_:
                vote+=i
            if(vote>0):
                l.append(1)
            else:
                l.append(0)
        return l
    def predict(self,X):
        out_of_fold=np.zeros((X.shape[0],len(self.basemodels)))
        for i,basemodels in enumerate(self.basemodels_):
            prediction=np.zeros((X.shape[0],len(basemodels)))
            for j,model in enumerate(basemodels):
                prediction[:,j]=model.predict(X)
            out_of_fold[:,i]=self.__vote(prediction)
        return self.metamodel_.predict(out_of_fold)  
if(__name__=="__main__"):
    modelTest2()