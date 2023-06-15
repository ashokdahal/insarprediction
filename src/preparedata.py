import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
class readInsarData():
    def __init__(self,dataprepinargs):
        self.dataparam=dataprepinargs
        self.wd=dataprepinargs['workingdir']
        self.filterdata=dataprepinargs['filterdata']
        self.filtercolumn=dataprepinargs['filtercolumn']
        self.filtervalue=dataprepinargs['filtervalue']
        self.testfiltervalue=dataprepinargs['testfiltervalue']
        
    def readfiles(self,filenm,filtervalue):
        covars=pd.read_csv(f"{self.wd+filenm}",sep=',') 
        covars[self.dataparam['catcols']]=covars[self.dataparam['catcols']].astype(str)
        if self.filterdata==True:
            return covars.loc[covars[self.filtercolumn]==filtervalue]
        else:
            return covars
    
    def preparedata(self):
       
        self.covars=self.readfiles(self.dataparam['filename'],filtervalue=self.filtervalue)
        numericalvars=self.covars[self.dataparam['numericalcols']]
        catvars=pd.get_dummies(self.covars[self.dataparam['catcols']])
        # print(catvars)
        # print(catvars.shape,numericalvars.sahpe)
        if self.dataparam['includecat']:
                Xtrain=np.hstack([numericalvars,catvars])
        else:
                Xtrain=numericalvars
        

        if self.dataparam['logarthmicY']:
            Ytrain=np.log(self.covars[self.dataparam['targetcol']].to_numpy()+self.dataparam['targetoffset'])
        if self.dataparam['onlyoffsetY']:
            Ytrain=self.covars[self.dataparam['targetcol']].to_numpy()+self.dataparam['targetoffset']
        if self.dataparam['notransformY']:
            Ytrain=self.covars[self.dataparam['targetcol']].to_numpy()
        if self.dataparam['absY']:
            Ytrain=np.abs(self.covars[self.dataparam['targetcol']].to_numpy())
        if self.dataparam['abslogY']:
            Ytrain=np.log(np.abs(self.covars[self.dataparam['targetcol']].to_numpy())+1)
        # print(Xtrain.shape)

        if self.dataparam['testbyfile']:
            self.testcovars=self.readfiles(self.dataparam['testfilename'],filtervalue=self.testfiltervalue)
            numericalvars=self.testcovars[self.dataparam['numericalcols']]
            catvars=pd.get_dummies(self.testcovars[self.dataparam['catcols']])
            # print(catvars.shape,numericalvars.sahpe)
            if self.dataparam['includecat']:
                Xtest=np.hstack([numericalvars,catvars])
            else:
                Xtest=numericalvars#np.hstack([numericalvars,catvars])
            if self.dataparam['logarthmicY']:
                Ytest=np.log(self.testcovars[self.dataparam['targetcol']].to_numpy()+self.dataparam['targetoffset'])
            if self.dataparam['onlyoffsetY']:
                Ytest=self.testcovars[self.dataparam['targetcol']].to_numpy()+self.dataparam['targetoffset']
            if self.dataparam['notransformY']:
                Ytest=self.testcovars[self.dataparam['targetcol']].to_numpy()
            if self.dataparam['absY']:
                Ytest=np.abs(self.testcovars[self.dataparam['targetcol']].to_numpy())
            if self.dataparam['abslogY']:
                Ytest=np.log(np.abs(self.testcovars[self.dataparam['targetcol']].to_numpy())+1)
            mins= np.min(np.hstack([Xtrain.min(axis=0),Xtest.min(axis=0)]),axis=0)
            maxs= np.max(np.hstack([Xtrain.max(axis=0),Xtest.max(axis=0)]),axis=0)
            Xtest=((Xtest - mins) / (maxs - mins))
            Xtrain=((Xtrain - mins) / (maxs - mins))
            self.Xtrain,self.Ytrain,self.Xtest,self.Ytest=Xtrain,Ytrain,Xtest,Ytest
        else:
            mins= np.min(Xtrain,axis=0)
            maxs= np.max(Xtrain,axis=0)
            Xtrain=((Xtrain - mins) / (maxs - mins))
            self.Xtrain,self.Xtest,self.Ytrain,self.Ytest=train_test_split(Xtrain, Ytrain, test_size=self.dataparam['testsize'], random_state=420)