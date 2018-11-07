# -*- coding: utf-8 -*-
import sys
import numpy as np
import re
import os 
import pandas as pd
from scipy.stats import uniform
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

'''
this module is for parsing data
'''
cwd=os.getcwd()
trainData_path=os.path.join(os.path.dirname(cwd),'data/train.csv')
testData_path=os.path.join(os.path.dirname(cwd),'data/test.csv')

def parseRawData2Numeric():
    '''
    this func is for turning raw data to numerical data
    including missing handling ,one hot encoding
    here is a rule when we process raw ,we try do not alter the original data
    we formulate  a new column
    '''
    train=pd.read_csv(trainData_path,header=0,index_col=None)
    test=pd.read_csv(testData_path,header=0,index_col=None)
    train_test={'train':train,'test':test}
    
    #one-hot encoding sex
    sex_dict={'female':0,'male':1}
    for key in train_test:
        logging.info('Sex  encoding {}'.format(key))
        data=train_test[key]
        data['_Sex_']=data['Sex'].map(sex_dict)
        logging.info('Sex  encoding {} is done'.format(key))
    
    #fill the nan embarked in train,all this should be explored ahead
    logging.info('--------------handling embarked miss in train -----------')
    embarked_na_index=pd.isna(train['Embarked'])
    logging.info('{} missing Embarked records'.format(np.sum(embarked_na_index)))
    train.loc[embarked_na_index,'Embarked']='S'
    logging.info('Embarked missing handling is done. {} miss remain(s)'.format(np.sum(pd.isna(train['Embarked']))))
    
    #one-hot encoding Embarked
    Embarked_dict={'S':0,'C':1,'Q':2}
    for key in train_test:
        logging.info('------- {}: encoding Embarked------'.format(key))
        data=train_test[key]
        data['_Embarked_']=data['Embarked'].map(Embarked_dict)
        logging.info('{}: encoding Embarked is done'.format(key))
        
    # extract title from name
    pattern='Miss|Mrs|Master|Mr'
    def extract_title(data):
        titles=[]
        for index in data.index:
            title=re.search(pattern,data[index])
            if(title!=None):
                titles.append(title.group())
            else:
                titles.append("Rare")
        return titles
    #extract titles
    for key in train_test:
        logging.info('Extracting titles from {}'.format(key))
        data=train_test[key]
        titles=extract_title(data['Name'])
        data['Title']=titles
        logging.info('Extract titles from {} is done'.format(key))
    
    #encoding Titles
    titles_dict={'Mr':0,'Mrs':1,'Miss':2,'Rare':3,'Master':4}
    for key in train_test:
        logging.info('encoding Title in {}'.format(key))
        data=train_test[key]
        data['_Title_']=data['Title'].map(titles_dict)
        logging.info('encoding titles in {} is done'.format(key))
        
    # fill the nan Age in train
    """
    we group the age by title,and we random age from mean+/- std
    """
    for key in train_test:
        data=train_test[key]
        mean_age_devided_by_title=data[['Title','Age']].groupby(by=['Title']).mean()
        std_age_devided_by_title=data[['Title','Age']].groupby(by=['Title']).std()
        bottom=mean_age_devided_by_title-std_age_devided_by_title
        upper=mean_age_devided_by_title+std_age_devided_by_title
        nan_index=data['Age'].isna()
        logging.info('random a age to fill the nan in {}'.format(key))
        for index in data[nan_index].index:
            title_=data.loc[index,'Title']
            random_age=uniform.rvs(bottom.loc[title_],upper.loc[title_])
            data.loc[index,'Age']=int(random_age)
        logging.info('all nan age in {} should be filled'.format(key))
        logging.info('{1} nan age value in {0}'.format(key,data['Age'].isna().sum()))
    
    #fill fare nan value in test
    fare_nan_index=test[pd.isna(test['Fare'])].index
    test.loc[fare_nan_index,"Fare"]=7.5
    
    #extract HasCabin feature from Cabin
    for key in train_test:
        logging.info('extract HasCabin from {}'.format(key))
        data=train_test[key]
        data['_HasCabin_']=0
        cabin_not_nan_index=data['Cabin'].notna()
        data.loc[cabin_not_nan_index,'_HasCabin_']=1
        logging.info('extract HasCabin from {} is done'.format(key))
    
    features_in_train=['Survived','Pclass','_Title_','_Sex_','SibSp','Parch','Fare','Age','_HasCabin_','_Embarked_']
    features_in_test=['Pclass','_Title_','_Sex_','SibSp','Parch','Fare','Age','_HasCabin_','_Embarked_']
    
    train_write_to_path=os.path.join(os.path.dirname(cwd),'data/train_.csv')
    test_write_to_path=os.path.join(os.path.dirname(cwd),'data/test_.csv')
    train[features_in_train].to_csv(train_write_to_path)
    test[features_in_test].to_csv(test_write_to_path)
    logging.info('successfully write train_,test_ to disk')

if(__name__=='__main__'):
    parseRawData2Numeric()