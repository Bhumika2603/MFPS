
def load_model(model_name):
    
    if model_name=='svm':
        from sklearn.svm import SVC
        return SVC(kernel='linear',random_state=0,probability=True)
    
    elif  model_name=='dt':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state = 0)
    
    elif model_name=='nb':
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    
    elif model_name=='cb':
        from catboost import CatBoostClassifier
        params = {'learning_rate': 0.1, 'depth': 6, 
                'l2_leaf_reg': 3, 'iterations': 100} 
        return CatBoostClassifier(**params)
    
    elif model_name=='rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()
    
    elif model_name=='xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(objective='binary:logistic') 
    
    elif model_name=='knn':
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=4)
    
    else:
        raise Exception("We does not support {} algo till now".format(model_name))
