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
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
cwd=os.getcwd()
train_data_path=os.path.join(os.path.dirname(cwd),'data/train_.csv')
test_data_path=(os.path.dirname(cwd),'data/test_.csv')

# FamilySize feature :SibSp + Parch
def getFamilySize(DataFrame):
    logging.info('starting to extract familysize')
    familysize=DataFrame['SibSp']+DataFrame['Parch']
    DataFrame['FamilySize']=familysize
    logging.info('extraction is done')
    return DataFrame


    
    