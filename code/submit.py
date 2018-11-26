import ModelFit as MF
import FeatureExtraction as FE
import os 
import logging
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,KFold
from sklearn.base import BaseEstimator,ClassifierMixin,TransformerMixin,clone
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
import xgboost as xgb
import warnings 
import numpy as np
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
train_path=os.path.join(os.path.dirname(os.getcwd()),'data/train_.csv')
test_path=os.path.join(os.path.dirname(os.getcwd()),'data/test_.csv')

def submit1():
    train=pd.read_csv(train_path,header=0,index_col=0)
    test=pd.read_csv(test_path,header=0,index_col=0)
    train_test={'train':train,'test':test}
    for key in train_test:
        data=train_test[key]
        logging.info('------getting FamilySize feature for {}-----'.format(key))
        data['Family_Size']=FE.getFamilySize(data)
        logging.info('-------done getting FamilySize feature for {} -----'.format(key))
        logging.info('------getting IsAlone feature for {}-----'.format(key))
        data['IsAlone']=FE.getIsAlone(data)
        logging.info('------done getting IsAlone feature for {}------'.format(key))
    y=train.pop("Survived").values
    X=train.values
    rf=RandomForestClassifier(100)
    bg=GradientBoostingClassifier(n_estimators=100)
    xg=xgb.XGBClassifier(n_estimators=100)
    tree=DecisionTreeClassifier()
    stacking=MF.StackingModel([rf,bg,xg],tree,cv=5)
    stacking.fit(X,y)
    my_prediction=stacking.predict(test.values)
    passengerId=np.arange(892,892+418)
    prediction_path=os.path.join(os.path.dirname(os.getcwd()),'data/gender_submission.csv ')
    pd.DataFrame({'PassengerId':passengerId,'Survived':my_prediction}).to_csv(prediction_path,index=False)
    logging.info('Successfully done')
    
if(__name__=="__main__"):
    submit1()    
