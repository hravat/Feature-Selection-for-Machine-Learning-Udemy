import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
import seaborn as sns
import statsmodels.stats.proportion as smprop
from sklearn import preprocessing
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression

class FeatureSelectionMutualInfo():
    '''
     Class used to check and select Featues based on Mutual Inforation
     The higher the mutual info value the better it is in predicting target
    '''
    
    def __init__(self,data_path):
        self.df=pd.read_csv(data_path)
        
    def dataframe_info(self):
        'Gives the number of records and columns and the size of the dataframe'
        self.df.info()
        
    def dataframe_stats(self):
        'Gives the summary statistics like perentieles means std deviations etc.'
        print(self.df.describe().T)
        
        
    def __train_test_split_fmb(self,target):
        '''
        This is a private function
        Pass in a data frame it it will split it in test and training sets
        Arguments :- target column to be predicted
        '''
        
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop([target],axis='columns'),\
                                                            self.df[target],\
                                                            test_size=0.3, \
                                                            random_state = 0
                                                           )
        return  X_train, X_test, y_train, y_test
    
    
    def __fit_summary(self,mi,initial_shape_xtrain,initial_shape_xtest,after_shape_xtrain,after_shape_xtest,retained_features,X_train_initial):        
        '''
        This is a private function
        It prints the summary
        '''
            
        print('----Fit Summary------')
        print('---------------------')
            
        print('The train shape before fit is '+str(initial_shape_xtrain))
        print('The train shape after fit is '+str(after_shape_xtrain))
        print('The test shape before fit is '+str(initial_shape_xtest))
        print('The test shape after fit is '+str(after_shape_xtrain))
            
        print('The below features are retained')
        print(retained_features)
            
        mi = pd.Series(mi)
        mi.index = X_train_initial.columns
        mi.sort_values(ascending=False,inplace=True)
        
        mi=pd.DataFrame(mi)
        
        plt.figure(figsize=(25,25))
        sns.heatmap(mi,cmap='Blues',annot = True , linewidths=4)
   
    def mutual_info(self,target,m_type='classification',select_style='kbest',select_value=10):
        '''
         This function gives out and displays summary of mutual information
         from scikit learn
         It requires target variable of the data dataframe to be specified
         select_style arguemnt takes 2 values kbest,percentile
        '''
        X_train, X_test, y_train, y_test = self.__train_test_split_fmb(target)
        initial_shape_xtrain = np.shape(X_train)
        initial_shape_xtest = np.shape(X_test)
        
        X_train_initial=X_train
        
        if m_type=='classification':
            mi = mutual_info_classif(X_train, y_train)
        else:
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numerical_vars = list(X_train.select_dtypes(include=numerics).columns)
            X_train=X_train[numerical_vars]
            X_test=X_test[numerical_vars]
            X_test.fillna(0,inplace=True)
            X_train.fillna(0,inplace=True)
            X_train_initial=X_train
            mi = mutual_info_regression(X_train, y_train)
            
        
        if select_style=='kbest':
            sel_ = SelectKBest(mutual_info_classif, k=select_value).fit(X_train, y_train)
        else:
            sel_ = SelectPercentile(mutual_info_classif, percentile=select_value).fit(X_train, y_train)
        
        retained_features = X_train.columns[sel_.get_support()]
        
        X_train = sel_.transform(X_train)
        X_test = sel_.transform(X_test)
        
        after_shape_xtrain = np.shape(X_train)
        after_shape_xtest = np.shape(X_test)
        
        self.__fit_summary(mi,initial_shape_xtrain,initial_shape_xtest,
                      after_shape_xtrain,after_shape_xtest,retained_features,X_train_initial)
        
        mi = pd.Series(mi)
        mi.index = X_train_initial.columns
        mi.sort_values(ascending=False,inplace=False)
        
        
        return mi

class FeatureSelectionChiSquare():
        '''
         Class used to check and select Featues based on Chi Square
        '''

        def __init__(self, data_path):
            self.df = pd.read_csv(data_path)

        def dataframe_info(self):
            'Gives the number of records and columns and the size of the dataframe'
            self.df.info()

        def dataframe_stats(self):
            'Gives the summary statistics like perentieles means std deviations etc.'
            print(self.df.describe().T)

        def __train_test_split_fmb(self, target):
            '''
            This is a private function
            Pass in a data frame it it will split it in test and training sets
            Arguments :- target column to be predicted
            '''

            X_train, X_test, y_train, y_test = train_test_split(self.df.drop([target], axis='columns'), \
                                                                self.df[target], \
                                                                test_size=0.3, \
                                                                random_state=0
                                                                )
            return X_train, X_test, y_train, y_test

        def chi_square_feature_select(self,column_list,target,select_style='kbest',select_value=10):
            '''
        Select feateures based on the Chi Square Test
        '''

            X_train, X_test, y_train, y_test = self.__train_test_split_fmb(target)

            for col in column_list:

                imputation_value = pd.DataFrame(X_train[col].value_counts(ascending=False,dropna=True))
                X_train[col].fillna(imputation_value.index[0], inplace=True)
                le=preprocessing.LabelEncoder()
                col_chi=le.fit_transform(X_train[col])
                model_summary = chi2(np.expand_dims(col_chi, axis=-1), y_train)

                print('----------------------------------------')
                print('The summary for '+col+' is given below')
                print('The f-score is '+str(round(float(model_summary[0]),3)))
                print('The p-value is ' + str(float(model_summary[1])))

class FeatureSelectionAnova():
    '''
             Class used to check and select Featues based on Anova
    '''

    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

    def dataframe_info(self):
        'Gives the number of records and columns and the size of the dataframe'
        self.df.info()

    def dataframe_stats(self):
        'Gives the summary statistics like perentieles means std deviations etc.'
        print(self.df.describe().T)

    def __train_test_split_fmb(self, target):
        '''
        This is a private function
        Pass in a data frame it it will split it in test and training sets
        Arguments :- target column to be predicted
        '''

        X_train, X_test, y_train, y_test = train_test_split(self.df.drop([target], axis='columns'), \
                                                            self.df[target], \
                                                            test_size=0.3, \
                                                            random_state=0
                                                            )
        return X_train, X_test, y_train, y_test

    def __fit_summary(self, univariate , initial_shape_xtrain, initial_shape_xtest, after_shape_xtrain, after_shape_xtest,
                      retained_features, X_train_initial):
        '''
        This is a private function
        It prints the summary
        '''

        print('----Fit Summary------')
        print('---------------------')

        print('The train shape before fit is ' + str(initial_shape_xtrain))
        print('The train shape after fit is ' + str(after_shape_xtrain))
        print('The test shape before fit is ' + str(initial_shape_xtest))
        print('The test shape after fit is ' + str(after_shape_xtrain))

        print('The below features are retained')
        print(retained_features)

        univariate = pd.Series(univariate[1])
        univariate.index = X_train_initial.columns
        univariate = univariate.sort_values(ascending=False)

        sns.set_style('whitegrid')
        plt.figure(figsize=(10, 10))
        plt.xticks(rotation=90)
        sns.barplot(x=univariate.index, y=univariate)
        plt.show()



    def anova(self, target, m_type='classification', select_style='kbest', select_value=10):
        '''
         This function gives out and displays summary of mutual information
         from scikit learn
         It requires target variable of the data dataframe to be specified
         select_style arguemnt takes 2 values kbest,percentile
        '''
        X_train, X_test, y_train, y_test = self.__train_test_split_fmb(target)
        initial_shape_xtrain = np.shape(X_train)
        initial_shape_xtest = np.shape(X_test)

        X_train_initial = X_train

        if m_type == 'classification':
            univariate = f_classif(X_train, y_train)
            f_select=f_classif
        else:
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numerical_vars = list(X_train.select_dtypes(include=numerics).columns)
            X_train = X_train[numerical_vars]
            X_test = X_test[numerical_vars]
            X_test.fillna(0, inplace=True)
            X_train.fillna(0, inplace=True)
            X_train_initial = X_train
            univariate = f_regression(X_train.fillna(0), y_train)
            f_select = f_regression



        if select_style == 'kbest':
            sel_ = SelectKBest(f_select, k=select_value).fit(X_train, y_train)
        else:
            sel_ = SelectPercentile(f_select, percentile=select_value).fit(X_train, y_train)

        retained_features = X_train.columns[sel_.get_support()]

        X_train = sel_.transform(X_train)
        X_test = sel_.transform(X_test)

        after_shape_xtrain = np.shape(X_train)
        after_shape_xtest = np.shape(X_test)

        self.__fit_summary(univariate, initial_shape_xtrain, initial_shape_xtest,
                           after_shape_xtrain, after_shape_xtest, retained_features, X_train_initial)

        univariate = pd.Series(univariate[1])
        univariate.index = X_train_initial.columns
        univariate = univariate.sort_values(ascending=False)

        return univariate














    
    
    

