import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import compress
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


class FilterMethodConstantFeatures:
    def __init__(self,data_path):
        self.df=pd.read_csv(data_path)
        
    def dataframe_info(self):
        self.df.info()
        
    def dataframe_stats(self):
        print(self.df.describe().T)
        
        
    def __train_test_split_fmb(self,target):
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop([target],axis='columns'),\
                                                            self.df[target],\
                                                            test_size=0.3, \
                                                            random_state = 0
                                                           )
        return  X_train, X_test, y_train, y_test
    
    def __fit_summary(self,num_not_constant_features,num_constant_features,initial_shape_xtrain,initial_shape_xtest,X_train,X_test):        
            print('----Fit Summary------')
            print('---------------------')
            print('A total of '+str(num_not_constant_features)+' features are not constant')
            print('A total of '+str(num_constant_features)+' features are constant')
            print('The train shape before fit is '+str(initial_shape_xtrain))
            print('The train shape after fit is '+str(np.shape(X_test)))
            print('The test shape before fit is '+str(initial_shape_xtest))
            print('The test shape after fit is '+str(np.shape(X_train)))
    
    
            
        
    def varinance_threshold(self,target):
        X_train, X_test, y_train, y_test = self.__train_test_split_fmb(target)
        
        sel = VarianceThreshold(threshold=0)
        sel.fit(X_train)  # fit finds the features with zero variance
        
        initial_shape_xtrain = np.shape(X_train)
        initial_shape_xtest = np.shape(X_test)
        
        X_train = sel.transform(X_train)
        X_test = sel.transform(X_test)
        
        # get_support is a boolean vector that indicates which features are retained
        # if we sum over get_support, we get the number of features that are not constant 
        # True os constant False is not constant
        
        self.__fit_summary(np.sum(sel.get_support()),
                           np.sum(~sel.get_support()),
                           initial_shape_xtrain,
                           initial_shape_xtest,
                           X_train,
                           X_test)
        
        
        
    def pandas_std(self,target):
        X_train, X_test, y_train, y_test = self.__train_test_split_fmb(target)   
        consant_features =[features for features in X_train.columns if X_train[features].std() == 0 ]
        
        initial_shape_xtrain = np.shape(X_train)
        initial_shape_xtest = np.shape(X_test)
        initial_num_features = len(X_train.columns)
        
        
        X_train.drop(consant_features,inplace=True,axis=1)
        X_test.drop(consant_features,inplace=True,axis=1)
        after_num_features = len(X_train.columns) 
        
        self.__fit_summary(after_num_features,
                           initial_num_features-after_num_features,
                           initial_shape_xtrain,
                           initial_shape_xtest,
                           X_train,
                           X_test)
        
    def pandas_nunique(self,target):
        X_train, X_test, y_train, y_test = self.__train_test_split_fmb(target)   
        consant_features =[features for features in X_train.columns if X_train[features].nunique() == 1]
        
        initial_shape_xtrain = np.shape(X_train)
        initial_shape_xtest = np.shape(X_test)
        initial_num_features = len(X_train.columns)
        
        
        X_train.drop(consant_features,inplace=True,axis=1)
        X_test.drop(consant_features,inplace=True,axis=1)
        after_num_features = len(X_train.columns) 
        
        self.__fit_summary(after_num_features,
                           initial_num_features-after_num_features,
                           initial_shape_xtrain,
                           initial_shape_xtest,
                           X_train,
                           X_test)
        
        
class FilterMethodQuasiConstantFeatures:
    
    def __init__(self,data_path):
        self.df=pd.read_csv(data_path)
        
    def dataframe_info(self):
        self.df.info()
        
    def dataframe_stats(self):
        print(self.df.describe().T)
        
    def __train_test_split_fmb(self,target):
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop([target],axis='columns'),\
                                                            self.df[target],\
                                                            test_size=0.3, \
                                                            random_state = 0
                                                           )
        return  X_train, X_test, y_train, y_test
    
    def __fit_summary(self,num_not_constant_features,num_constant_features,initial_shape_xtrain,initial_shape_xtest,X_train,X_test,var_threshold=0):        
            print('----Fit Summary------')
            print('---------------------')
            if var_threshold == 0:
                print('A total of '+str(num_not_constant_features)+' features are not constant')
                print('A total of '+str(num_constant_features)+' features are constant')
            else:
                print('A variance threshold  of '+str(var_threshold)+' is used')
                print('A total of '+str(num_not_constant_features)+' features are not constant')
                print('A total of '+str(num_constant_features)+' features are quasi constant')            
            print('The train shape before fit is '+str(initial_shape_xtrain))
            print('The train shape after fit is '+str(np.shape(X_test)))
            print('The test shape before fit is '+str(initial_shape_xtest))
            print('The test shape after fit is '+str(np.shape(X_train)))
            
    def __quasi_variance_summary(self,df_X_train,col_names):
        
        
        print('--------Summary of Value Percenteges--------------')
        
        for col in col_names:
            percent=round(df_X_train[col].value_counts()[0] / np.float(len(df_X_train))*100,4)
            value=df_X_train[col].value_counts().index[0]
            print("For column "+col+" the value "+str(value)+" has "+str(percent)+" percent of values")
        
        
        
            
            
            
    def pandas_nunique_constant_features(self,target,print_summ=True):
        X_train, X_test, y_train, y_test = self.__train_test_split_fmb(target)   
        consant_features =[features for features in X_train.columns if X_train[features].nunique() == 1]
        
        initial_shape_xtrain = np.shape(X_train)
        initial_shape_xtest = np.shape(X_test)
        initial_num_features = len(X_train.columns)
        
        
        X_train.drop(consant_features,inplace=True,axis=1)
        X_test.drop(consant_features,inplace=True,axis=1)
        after_num_features = len(X_train.columns) 
        
        if print_summ:
            self.__fit_summary(after_num_features,
                               initial_num_features-after_num_features,
                               initial_shape_xtrain,
                               initial_shape_xtest,
                               X_train,
                               X_test)        
        if print_summ==False:
            return X_train, X_test, y_train, y_test
        
       
      
    
    
    
    def varinance_threshold(self,var_threshold,target):
        
        X_train, X_test, y_train, y_test = self.pandas_nunique_constant_features(target,print_summ=False)  
        col_names=X_train.columns
        # Make deep instead of shallow copy to create an entirly new datafeame
        df_X_train=X_train.copy(deep=True)
        sel = VarianceThreshold(threshold=var_threshold)
        sel.fit(X_train)  # fit finds the features with zero variance
        
        initial_shape_xtrain = np.shape(X_train)
        initial_shape_xtest = np.shape(X_test)
        
        X_train = sel.transform(X_train)
        X_test = sel.transform(X_test)
        
        # get_support is a boolean vector that indicates which features are retained
        # if we sum over get_support, we get the number of features that are not constant 
        # True os constant False is not constant
        
        self.__fit_summary(np.sum(sel.get_support()),
                           np.sum(~sel.get_support()),
                           initial_shape_xtrain,
                           initial_shape_xtest,
                           X_train,
                           X_test,
                           var_threshold)        
        col_names = list(compress(col_names,~sel.get_support()))
        
        self.__quasi_variance_summary(df_X_train,col_names)
        
        
    def quasi_constant_manual(self,target,threshold=0.998):
            
        X_train, X_test, y_train, y_test = self.pandas_nunique_constant_features(target,print_summ=False)    
        col_names=X_train.columns
        # Make deep instead of shallow copy to create an entirly new datafeame
        df_X_train=X_train.copy(deep=True)
        quasi_constant_feat=[]    
        for col in col_names:
            # find the predominant value, that is the value that is shared
            # by most observations
            predominant = (X_train[col].value_counts() / np.float(
            len(X_train))).sort_values(ascending=False).values[0]

            # evaluate the predominant feature: do more than 99% of the observations
            # show 1 value?
            if predominant > threshold:
                # if yes, add the variable to the list
                quasi_constant_feat.append(col)
        
        initial_shape_xtrain = np.shape(X_train)
        initial_shape_xtest = np.shape(X_test)
        
        
        X_train.drop(labels=quasi_constant_feat, axis=1, inplace=True)
        X_test.drop(labels=quasi_constant_feat, axis=1, inplace=True)
            
       
          
        
        self.__fit_summary(len(X_train.columns)-len(quasi_constant_feat),
                           len(quasi_constant_feat),
                           initial_shape_xtrain,
                           initial_shape_xtest,
                           X_train,
                           X_test,
                           threshold)        
        
        self.__quasi_variance_summary(df_X_train,quasi_constant_feat)            
        
        
class FilterMethodDuplicateFeatures:
    
    def __init__(self,data_path):
        self.df=pd.read_csv(data_path)
        
    def dataframe_info(self):
        self.df.info()
        
    def dataframe_stats(self):
        print(self.df.describe().T)
        
    def __train_test_split_fmb(self,target):
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop([target],axis='columns'),\
                                                            self.df[target],\
                                                            test_size=0.3, \
                                                            random_state = 0
                                                           )
        return  X_train, X_test, y_train, y_test
    
    
    def __quasi_constant_manual(self,target,threshold=0.998):
        X_train, X_test, y_train, y_test = self.__train_test_split_fmb(target)     
        col_names=X_train.columns
        # Make deep instead of shallow copy to create an entirly new datafeame
        quasi_constant_feat=[]    
        for col in col_names:
            # find the predominant value, that is the value that is shared
            # by most observations
            predominant = (X_train[col].value_counts() / np.float(
            len(X_train))).sort_values(ascending=False).values[0]

            # evaluate the predominant feature: do more than 99% of the observations
            # show 1 value?
            if predominant > threshold:
                # if yes, add the variable to the list
                quasi_constant_feat.append(col)
        
        X_train.drop(labels=quasi_constant_feat, axis=1, inplace=True)
        X_test.drop(labels=quasi_constant_feat, axis=1, inplace=True)        
            
        return X_train,X_test
    
    
    def __fit_summary(self,dup_list,before_test_shape,after_test_shape,before_train_shape,after_train_shape):
        
        print('--------summary of duplicate features------------------')
        print('There are a total of '+str(len(dup_list))+" duplicate features")
        print('--------------------------------------------------------------')
        for l in dup_list:
            print(l[1]+" is a duplicate of "+l[0]+" and will be dropped")
        print('--------------------------------------------------------------')
        print('Train Shape before drop '+str(before_test_shape))
        print('Train Shape after drop '+str(after_test_shape))
        print('Test Shape before drop '+str(before_train_shape))
        print('Test Shape after drop '+str(after_train_shape))
    
    
    def duplicate_features(self,target):
        X_train,X_test = self.__quasi_constant_manual(target)
        
        dup_dict={}
        dup_list=[]
        col_list = X_train.columns
        drop_list = []
        
        for i in  range(len(X_train.columns)):
            for j in range(i,len(X_train.columns)):
                if (col_list[i] != col_list[j]) and (X_train.iloc[:,i].equals(X_train.iloc[:,j])):
                    dup_list.append([col_list[i],col_list[j]])
                    drop_list.append(col_list[j])
        
        before_train_shape = np.shape(X_train)
        before_test_shape  = np.shape(X_test)
        
        X_train.drop(drop_list,axis=1,inplace=True)  
        X_test.drop(drop_list,axis=1,inplace=True)
        
        after_train_shape = np.shape(X_train)
        after_test_shape = np.shape(X_test)
          
        self.__fit_summary(dup_list,before_test_shape,after_test_shape,before_train_shape,after_train_shape)   
        
        
        
class FilterMethodCorrelatedFeatues:
    
    def __init__(self,data_path):
        self.df=pd.read_csv(data_path)
        
    def dataframe_info(self):
        self.df.info()
        
    def dataframe_stats(self):
        print(self.df.describe().T)
        
    def __train_test_split_fmb(self,target):
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop([target],axis='columns'),\
                                                            self.df[target],\
                                                            test_size=0.3, \
                                                            random_state = 0
                                                           )
        return  X_train, X_test, y_train, y_test
    
    def __correlation_summary(self,summary_list,
                                   before_train_shape,
                                   before_test_shape,
                                   after_train_shape,
                                   after_test_shape):
        
        print('----------Correlated Features Summary---------------')
        print('There are a total of '+str(len(summary_list))+' correlated features')
        print('----------------------------------------------------')      
        print('Train Shape before drop '+str(before_test_shape))
        print('Train Shape after drop '+str(after_test_shape))
        print('Test Shape before drop '+str(before_train_shape))
        print('Test Shape after drop '+str(after_train_shape))
        print('----------------------------------------------------')
        
        
        for i in summary_list:
            print('The correlation beteen '+i[1]+' and '+i[2]+' is '+str(i[0]))
    
    def visualize_corr(self,target):
        X_train, X_test, y_train, y_test = self.__train_test_split_fmb(target)  
        
        correlation_matrix = X_train.corr(method='pearson')
        cmap = sns.diverging_palette(20, 220, as_cmap=True)

        # some more parameters for the figure
        fig, ax = plt.subplots()
        fig.set_size_inches(11,11)

        # and now plot the correlation matrix
        sns.heatmap(correlation_matrix, cmap=cmap)
        
        
    def correlation_brute_force(self,target,threshold=0.8):
    
    
        X_train, X_test, y_train, y_test = self.__train_test_split_fmb(target)
    
        col_corr = set()
        corr_matrix = X_train.corr()
        summary_list = []
    
    
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (abs(corr_matrix.iloc[i, j]) > threshold): 
                
                    summary_list.append([round(corr_matrix.iloc[i, j],3),corr_matrix.columns[i],corr_matrix.columns[j]])
                    # get the name of the correlated feature
                    colname = corr_matrix.columns[j]
                    # and add it to our correlated set
                    col_corr.add(colname)
                
                
        before_train_shape = np.shape(X_train)
        before_test_shape = np.shape(X_test)
        
        X_train.drop(col_corr,inplace=True,axis=1)
        X_test.drop(col_corr,inplace=True,axis=1)
        
        after_train_shape = np.shape(X_train)
        after_test_shape = np.shape(X_test)
                
        self.__correlation_summary(summary_list,
                                   before_train_shape,
                                   before_test_shape,
                                   after_train_shape,
                                   after_test_shape
                                  )
                
                
                
        return col_corr     
        
        
    def correlation_group_features(self,target,threshold=0.8):
            
        X_train, X_test, y_train, y_test = self.__train_test_split_fmb(target)
    
        col_corr = set()
        corr_matrix = X_train.corr()
        summary_list = []
    
    
        for i in range(len(corr_matrix.columns)):
            for j in range(i,len(corr_matrix.columns)):
                if (abs(corr_matrix.iloc[i, j]) > threshold) and (corr_matrix.columns[j] != corr_matrix.columns[i]): 
                    summary_list.append([corr_matrix.columns[i],corr_matrix.columns[j],round(abs(corr_matrix.iloc[i, j]),3)])
                    
        summary_list=pd.DataFrame(summary_list)
        summary_list.columns = ['feature1', 'feature2', 'corr']
        
        summary_list.sort_values(['feature1', 'feature2'])
        
        le = LabelEncoder()
        
        summary_list['group'] = le.fit_transform(summary_list['feature1'])
        
        
        return summary_list
            
            
            
            
            
            
       
    
    
    
    
    
    