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
train = Train_model()

# def predict(records):
#     df,cat_cols,df1_original= preprocess.preprocess_ev_data("../p.xlsx",dict["mean_value"],task)
#     df=preprocess.encode_ev(df,cat_cols,encoder)
#     # print(df.columns)
#     #df=preprocess.scale_ev_data(df)

#     classifier = dict['model']
#     y_pred_proba_test = train.evaluate(classifier,df)
#     y_predict=train.getPredUsingOptimalThrehold(dict['threshold'],y_pred_proba_test)
#     df1_original['PIH'] = y_predict
#     print("DATA IS")
#     print(df1_original)
#     return df1_original

df,cat_cols,df1_original= preprocess.preprocess_ev_data("../p.xlsx",dict["mean_value"],task)
df=preprocess.encode_ev(df,cat_cols,encoder)
# print(df.columns)
#df=preprocess.scale_ev_data(df)

classifier = dict['model']
y_pred_proba_test = train.evaluate(classifier,df)
y_predict=train.getPredUsingOptimalThrehold(dict['threshold'],y_pred_proba_test)
df1_original['PIH'] = y_predict
print("DATA IS")
print(df1_original)









