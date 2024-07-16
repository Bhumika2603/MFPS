#Importing required libraries
from preprocessing import Preprocessing
from train import Train_model
from plot import Plot
from metrices import Metrices
from hypertuning import optimize

import sys
from model import load_model
from utils import write_report,storing_model

#Making result directory
import os

base_dir = 'results/'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# algo =sys.argv[1]
algo  = 'cb'
task="task1"
encoding='OneHotEncoding'

# PREPROCESSING THE DATA
preprocess = Preprocessing()

df1,means = preprocess.preprocess_data(dataset_path)
# y = df1['PIH']
print(df1['PIH'])

# df1=df1.drop(['PIH'],axis=1)


if task=="task2.1": 
    task_ = 'task2.1/'
    X_train1,y_train1,X_test,y_test,X_val,y_val= preprocess.split_data(df1,'Encoded_ANC2')

elif task=="task2.2":
    task_ = 'task2.2/'
    X_train1,y_train1,X_test,y_test,X_val,y_val= preprocess.split_data(df1,'Encoded_ANC3')

elif task=="task2.3":
    task_ = 'task2.3/'
    X_train1,y_train1,X_test,y_test,X_val,y_val= preprocess.split_data(df1,'Encoded_ANC4')

else :
    task_ = 'task1/'
    # X=df1.drop(['PIH'],axis=1)
    X_train1,y_train1,X_test,y_test,X_val,y_val= preprocess.split_data(df1,'PIH')

# X_train1,X_val,X_test = preprocess.scale_data(X_train1,X_val,X_test)
algo_path=storing_model(base_dir,task_,algo)

#Loading and training the model
classifier = load_model(algo)
train = Train_model()
classifier,y_pred_proba_train,y_pred_proba_val = train.train(classifier,X_train1, y_train1,X_val)
y_pred_proba_test = train.evaluate(classifier,X_test)
y_predict_test,optimal_threshold = train.getPredictionsUsingThreshold(y_val,y_pred_proba_val,y_pred_proba_test)

dict={f"model":classifier,"mean_value":means,f"threshold":optimal_threshold}    
with open("model1.pkl","wb") as mod:
    import pickle
    pickle.dump(dict,mod)
    print("Model saved successfully!")

results = Metrices()
result_dict=results.get_and_printResults(y_test,y_predict_test)
write_report(result_dict,algo_path)

#plotting and storing plots
plots=Plot()
plots.plottingPR_ROC_ConfusionMatrix(algo,y_train1,y_val,y_test,y_pred_proba_train,y_pred_proba_val,y_pred_proba_test,y_predict_test,algo_path)







