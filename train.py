import numpy as np
from sklearn.metrics import precision_recall_curve 


class Train_model:
    def __init__(self):
        pass

    def train(self,model,X_train1, y_train1,X_val):

        model.fit(X_train1, y_train1)
        y_pred_proba_train = model.predict_proba(X_train1)[:,1]
        y_pred_proba_val = model.predict_proba(X_val)[:,1]

        
        return model,y_pred_proba_train,y_pred_proba_val
        
    def evaluate(self,model,X_test):
        y_pred_proba_test = model.predict_proba(X_test)[:,1]
        return y_pred_proba_test
    

    def getPredictionsUsingThreshold(self,y_actual,y_pred_prob_val,y_pred_prob_test):
        precision, recall, thresholds = precision_recall_curve(y_actual,y_pred_prob_val)
        optimal_threshold = self.getThresholdFromPRcurve(precision,recall,thresholds)
        y_predict = self.getPredUsingOptimalThrehold(optimal_threshold,y_pred_prob_test)
    
        return y_predict,optimal_threshold   


    def getThresholdFromPRcurve(self,precision,recall,thresholds):
        f1_scores = 2 * (precision * recall) / (precision+ recall)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        return optimal_threshold


    def getPredUsingOptimalThrehold(self,optimal_threshold,y_pred_proba):
        y_predict_class = [1 if prob > optimal_threshold else 0 for prob in y_pred_proba]
        
        return  y_predict_class


    