from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

class Metrices:
     def __init__(self):
        pass
     
     def get_and_printResults(self,y_test,y_predict_test):
        result_dict = {}
        result_dict["Accuracy"],result_dict["ConfusionMatrix"],result_dict["ClassificationReport"] = self.getAccuracyCmReport(y_test,y_predict_test)
        result_dict["Prevalence"],result_dict["Sensitivity"],result_dict["Specificity"] = self.getPrevalence_sensitivity_specificity(result_dict["ConfusionMatrix"])
        self.printResults(result_dict)
        return result_dict
     
     def printResults(self,result_dict):
        for k, v in result_dict.items():
            if k == "ClassificationReport" or k == "ConfusionMatrix":
                print(f'\n{k} : \n\n{v}')
            else:
                print(f'\n{k} : {v}')
     
     def getAccuracyCmReport(self,y_actual,y_pred):
        accuracy = accuracy_score(y_actual,y_pred)
        cm = self.getConfusionMatrix(y_actual, y_pred)
        report = self.getClassificationReport(y_actual,y_pred)

        return accuracy,cm,report

     def getConfusionMatrix(self,y_actual,y_pred):
        return confusion_matrix(y_actual, y_pred)
        
     def getClassificationReport(self,y_actual,y_pred):
        target_names = ['class 0', 'class 1']
        cr = classification_report(y_actual,y_pred,target_names=target_names)
        return cr


     def getPrevalence_sensitivity_specificity(self,cm):
        TP,TN,FP,FN = self.getpostivesNegatives(cm)
        Prevalence = self.getPrevalance(TP,TN,FP,FN)
        Sensitivity = self.getSensitivity(TP,FN)
        Specificity = self.getSpecificity(TN,FP)
        return Prevalence,Sensitivity,Specificity
    
     def getpostivesNegatives(self,cm):
        TP = cm[1,1]
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]
        
        return TP,TN,FP,FN
    
     def getPrevalance(self,TP,TN,FP,FN):
        prevalence = (FN+TP)/(TN+FP+FN+TP)
        return prevalence
        
     def getSensitivity(self,TP,FN):
        sensitivity = TP/(TP+FN)
        
        return sensitivity

     def getSpecificity(self,TN,FP):
        specificity = TN/(TN+FP)
        
        return specificity
        
            
      