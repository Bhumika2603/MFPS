import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_curve,ConfusionMatrixDisplay

# from main import algo_path
class Plot: 
    def __init__(self):
        pass

    def plottingPR_ROC_ConfusionMatrix(self,algo,y_train1,y_val,y_test,y_pred_proba_train,y_pred_proba_val,y_pred_proba_test,y_predict_test,algo_path):
       self.plot_PR(algo,y_train1,y_val,y_test,y_pred_proba_train,y_pred_proba_val,y_pred_proba_test,algo_path)
       self.plot_Roc(algo,y_train1,y_val,y_test,y_pred_proba_train,y_pred_proba_val,y_pred_proba_test,algo_path)
       self.plot_cm(algo,y_test,y_predict_test,algo_path)

    def plot_PR(self,model_name,y_train,y_val,y_test,y_pred_proba_train,y_pred_proba_val,y_pred_proba_test,algo_path):
        plt.figure(figsize=(8, 6))
        precision1, recall1, thresholds1 = precision_recall_curve(y_train,y_pred_proba_train) 
        precision2, recall2, thresholds2 = precision_recall_curve(y_val,y_pred_proba_val) 
        precision3, recall3, thresholds3 = precision_recall_curve(y_test,y_pred_proba_test)
        
        plt.plot(recall1, precision1,color='r',label='train')
        plt.plot(recall2, precision2,color='g',label='val')
        plt.plot(recall3, precision3,color='b',label='test')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve of {model_name} ')
        plt.legend()
        save_path = algo_path+'/pr.jpeg'
        plt.savefig(save_path)
        
        
    def plot_Roc(self,model_name,y_train,y_val,y_test,y_pred_proba_train,y_pred_proba_val,y_pred_proba_test,algo_path):
        plt.figure(figsize=(8, 6))
        print(y_pred_proba_train)
        fpr1, tpr1, thresh1 = roc_curve(y_train,y_pred_proba_train)
        fpr2, tpr2, thresh2 = roc_curve(y_val,y_pred_proba_val) 
        fpr3, tpr3, thresh3 = roc_curve(y_test,y_pred_proba_test)
        auc_score1 = roc_auc_score(y_train,y_pred_proba_train)
        auc_score2 = roc_auc_score(y_val,y_pred_proba_val)
        auc_score3 = roc_auc_score(y_test,y_pred_proba_test)

        plt.plot(fpr1, tpr1, label=f'train (AUC = {auc_score1:.2f})')
        plt.plot(fpr2, tpr2, label=f'val (AUC = {auc_score2:.2f})')
        plt.plot(fpr3, tpr3, label=f'test~ (AUC = {auc_score3:.2f})')

        plt.plot([0, 1], [0, 1], linestyle='--', color='black')  # Diagonal line for random classifier
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve of {model_name}')
        plt.legend()
        plt.grid(True)
        save_path = algo_path+'/roc.jpeg'
        plt.savefig(save_path)
        
    def plot_cm(self,model_name,y_actual,y_predict_class,algo_path):
        ConfusionMatrixDisplay.from_predictions(y_actual,y_predict_class )
        plt.title(f'{model_name} Confusion matrix ')
        save_path = algo_path+'/cm.jpeg'
        plt.savefig(save_path)
    