from preprocessing import Preprocessing
from train import Train_model
import sys
from model import load_model
import pickle

with open('encoder.pkl','rb') as enco:
    print("encoder load success")
    encoder=pickle.load(enco)

with open('model1.pkl', 'rb') as fp:
    dict=pickle.load(fp)

task="task1"
preprocess = Preprocessing() 
df,cat_cols= preprocess.preprocess_ev_data(r"C:\Users\hp\Downloads\p.xlsx",dict["mean_value"],task)
df=preprocess.encode_ev(df,cat_cols,encoder)
print(df.columns)
df=preprocess.scale_ev_data(df)

train = Train_model()
classifier = dict['model']
y_pred_proba_test = train.evaluate(classifier,df)
y_predict=train.getPredUsingOptimalThrehold(dict['threshold'],y_pred_proba_test)
print(y_predict)






