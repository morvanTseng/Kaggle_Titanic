# -*- coding: utf-8 -*-
'''
all function s in 
this module is to 
get training data for model
'''
import sys
import os 
import pandas as pd
import logging
import dataparse
from sklearn.preprocessing import LabelEncoder
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
cwd=os.getcwd()
train_data_path=os.path.join(os.path.dirname(cwd),'data/train_.csv')
test_data_path=(os.path.dirname(cwd),'data/test_.csv')
def onehot(df,*features):
    '''
    this func is to one hot encode
    '''
    df_=pd.DataFrame()
    for feature in features:
        logging.info('{}: getting dummies'.format(feature))
        dummies=pd.get_dummies(df[feature])
        df_.join(dummies)
        logging.info('{} :done dummies'.format(feature))
    logging.info('one hot is done')
# FamilySize feature :SibSp + Parch
def getFamilySize(DataFrame):
    logging.info('starting to extract familysize')
    familysize=DataFrame['SibSp']+DataFrame['Parch']
    logging.info('extraction is done')
    return familysize.tolist()
#IsAlone feature 
def getIsAlone(DataFrame):
    logging.info("-------get feature IsAlone--------")
    IsAlone=[]
    for index in DataFrame.index:
        if(DataFrame.loc[index,'Family_Size']==0):
            IsAlone.append(1)
        else:
            IsAlone.append(0)
            
    logging.info('---------done getting feature IsAlone-----------')
    return IsAlone  
    