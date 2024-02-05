import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from encoder import DataEncoder


class Preprocessing():
    def __init__(self):
        pass
        
    #LOADING DATA
    def load_data(self,path,task):
        df1 = pd.read_excel(path)

        if task=="task1":
         df1 = df1.dropna(subset = ['PIH'])
         df1 = df1.reset_index().drop('index',axis=1)
         df1 = self.dropnull(df1)
         df1=df1.drop(['PIH Mngt ','MUAC'],axis=1)
         return df1

        elif task=='task2.1':
            coln =  list(df1.loc[:,'ANC Date2':'MUAC'].columns)
            df1=df1.drop(coln,axis=1)
            df1 = df1.dropna(subset = ['ANC1'])
           
        
        elif task=="task2.2":
            coln =  list(df1.loc[:,'ANC Date3':'MUAC'].columns)
            df1=df1.drop(coln,axis=1)
            df1 = df1.dropna(subset = ['ANC2'])
        
        else:
            coln =  list(df1.loc[:,'ANC Date4':'MUAC'].columns)
            df1=df1.drop(coln,axis=1)
            df1 = df1.dropna(subset = ['ANC3'])
            
         
        df1 = df1.reset_index().drop('index',axis=1)
        df1 = self.dropnull(df1)
        return df1

    def process_date_colm(self,df1,cat_cols):
        df=pd.DataFrame()
        df1,df= self.handle_DateCols(df1,df)
        
        date_new_coln =  list(df.columns)
        cat_cols = list(cat_cols)+date_new_coln
        for i in date_new_coln:
            df1[i] = df1[i].astype(str)
        return df1, cat_cols

    def preprocess_data(self,path,task,encoding):
        df1 = self.load_data(path,task)
        df1,num_cols,cat_cols,means = self.impute(df1,task)
        df1,cat_cols = self.process_date_colm(df1,cat_cols)
        df1=self.encode_cat_cols(df1,cat_cols,task,encoding)
        return df1,means
    
    def preprocess_ev_data(self,path,mean_value,task):
        df1 = self.load_data(path,task)
        if task=="task1":
            df1=df1.drop(['PIH'],axis=1)

        elif task=="task2.1":
            df1=df1.drop(['ANC2'],axis=1)
        
        elif task=="task2.2":
            df1=df1.drop(['ANC3'],axis=1)       
            
        else:
             df1=df1.drop(['ANC4'],axis=1)   
       
        df1,num_cols,cat_cols = self.impute_ev(df1,mean_value,task)
        df1,cat_cols = self.process_date_colm(df1,cat_cols)
        return df1,cat_cols


    #DROPPING NULL COLUMNS
    def dropnull(self,df):
        l=[features for features in df.columns if df[features].isnull().sum()==len(df)]
        df = df.drop(l,axis=1)
        return df

    #IMPUTING

    def impute(self,data,task):
        cat_cols=data.select_dtypes(include=['object']).columns
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        means = {}
        for col in num_cols:
            if task=="task1":
                if col=='PIH':
                    continue
            means[col] = data[col].mean()
            data[col]=data[col].fillna(data[col].mean())
        
        #replacing missing values by unknown(only categorical variables)
        for col in cat_cols:
            if task=="task1":
               data[col]=data[col].fillna("unknown")
               return data,num_cols,cat_cols,means

            elif task=="task2.1":
                    if col=='ANC2':
                      data[col]=data[col].fillna("No")
            elif task=="task2.2":
                    if col=='ANC3':
                      data[col]=data[col].fillna("No")
            else:
                    if col=='ANC4':
                      data[col]=data[col].fillna("No")
            data[col]=data[col].fillna("unknown")

        return data,num_cols,cat_cols,means
    
    def impute_ev(self,data,mean_value,task):
        cat_cols=data.select_dtypes(include=['object']).columns
        num_cols = data.select_dtypes(include=np.number).columns.tolist()

        for col in num_cols:
            if task=="task1":
                if col=='PIH':
                    continue
            
            data[col]=data[col].fillna(mean_value)
        
        #replacing missing values by unknown(only categorical variables)
        for col in cat_cols:
            if task=="task1":
               data[col]=data[col].fillna("unknown")
               return data,num_cols,cat_cols

            elif task=="task2.1":
                    if col=='ANC2':
                      data[col]=data[col].fillna("No")
            elif task=="task2.2":
                    if col=='ANC3':
                      data[col]=data[col].fillna("No")
            else:
                    if col=='ANC4':
                      data[col]=data[col].fillna("No")
            data[col]=data[col].fillna("unknown")

        return data,num_cols,cat_cols 


    def handle_DateCols(self,data,df):
        columns=[column for column in data.columns if data[column].dtypes=='<M8[ns]']
        
        for cols in columns:
            df[f'{cols}_month']=data[cols].dt.month_name()
            df[f'{cols}_day']=data[cols].dt.day
        df=df.fillna("Unknown",axis=1)
        new_df = df.copy()
    
        combined_data = pd.concat([data, new_df], axis=1)
        combined_data = combined_data.drop(columns=columns, axis=1)
    
        return combined_data, df



    #ONE HOT ENCODING DATA
    def encode(self,data,cat_cols,encoding):
        dataEncoder = DataEncoder()
        data = dataEncoder.encode(data,cat_cols,encoding)
        return data
    
    def encode_ev(self,data,cat_cols,encoder):
        extra=pd.DataFrame()
        for col in cat_cols:
            results=encoder.transform(data[[col]])
            encoded_df=pd.DataFrame(results.toarray(),columns=encoder.get_feature_names_out())
            extra=pd.concat([extra,encoded_df],axis=1)
        combined_df = pd.concat([data,extra],axis=1)

        combined_df = combined_df.drop(cat_cols,axis=1)
        return combined_df
        
    def encode_cat_cols(self,df1,cat_cols,task,encoding):
        if task=="task1":
            df1 = self.encode(df1,cat_cols,encoding)
            return df1
        
        elif task=="task2.1":
            cat_cols = [i for i in cat_cols if i != 'ANC2']
            df1 = self.encode(df1,cat_cols,encoding)
            df1['Encoded_ANC2'] = df1['ANC2'].map({'Yes': 1, 'No': 0})
            df1=df1.drop('ANC2',axis=1)
        
        elif task=="task2.2":
            cat_cols = [i for i in cat_cols if i != 'ANC3']
            df1 = self.encode(df1,cat_cols,encoding)
            df1['Encoded_ANC3'] = df1['ANC3'].map({'Yes': 1, 'No': 0})
            df1=df1.drop('ANC3',axis=1)
        
        else:
            cat_cols = [i for i in cat_cols if i != 'ANC4']
            df1 = self.encode(df1,cat_cols,encoding)
            df1['Encoded_ANC4'] = df1['ANC4'].map({'Yes': 1, 'No': 0})
            df1=df1.drop('ANC4',axis=1)
        
        return df1
    
    
    #SPLITTING DATA


    def split_data(self,data,pred_column):
        # X=data.loc[:,data.columns!=pred_column]

        y= data[pred_column]
        X= data.drop(['PIH'],axis=1)


        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.1,train_size=0.9,
                                                        shuffle=True,stratify=data[pred_column])
        x1=X_train
        y1=y_train
        X_train1, X_val, y_train1, y_val = train_test_split(x1,y1,random_state=42,test_size=0.1,
                                                            train_size=0.9,shuffle=True,stratify=y1)
        
        return X_train1,y_train1,X_test,y_test,X_val,y_val
    
    # SCALING DATASETS

    def scale_data(self,X_train1,X_val,X_test):
        scaler=StandardScaler()
        scaler.fit_transform(X_train1)
        scaler.transform(X_val)
        scaler.transform(X_test)

        return X_train1,X_val,X_test
    
    def scale_ev_data(self,X_test):
        scaler=StandardScaler()
        scaler.fit_transform(X_test)

        return X_test
    
    


