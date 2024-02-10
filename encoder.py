from sklearn.preprocessing import OneHotEncoder
from category_encoders import count, cat_boost
import dirty_cat
import pandas as pd
class DataEncoder():

    def __init__(self):
        pass

    def encode(self,data,cat_cols,encoding):
        if encoding == 'OneHotEncoding':
            encoder = OneHotEncoder()

        elif encoding == 'NormalizedCountEncoding':
            encoder = count.CountEncoder(normalize=True)

        elif encoding == 'CatBoostEncoding':
            encoder = cat_boost.CatBoostEncoder(cols=cat_cols)

        elif encoding == 'SimilarityEncoding':
            encoder = dirty_cat.SimilarityEncoder(hashing_dim=5, categories='most_frequent', n_prototypes=10)

        elif encoding == "GapEncoding":
            encoder = dirty_cat.GapEncoder(n_components=5)

        elif encoding == "MinHashEncoding":
            encoder = dirty_cat.MinHashEncoder(n_components=1)
          
        else:
            raise (Exception(self.encoding + "hasn't been implemented yet"))
            
        if encoding == 'CatBoostEncoding' or encoding == 'NormalizedCountEncoding' :
            results=encoder.fit_transform(data[cat_cols],data['PIH'])
            encoded_df=pd.DataFrame(results,columns=encoder.get_feature_names_out())
            if encoding == 'CatBoostEncoding':
                encoded_df.columns = encoded_df.columns + "_CatBoost"
            else:
                encoded_df.columns = encoded_df.columns + "_NormCount"
            combined_df = pd.concat([data,encoded_df],axis=1)
            
        elif encoding=="GapEncoding" or encoding == 'SimilarityEncoding':
            results = encoder.fit_transform(data[cat_cols])
            encoded_df=pd.DataFrame(results,columns=encoder.get_feature_names_out())
            
            combined_df = pd.concat([data,encoded_df],axis=1)
        
        elif encoding=='MinHashEncoding':
            results = encoder.fit_transform(data[cat_cols])
            print(results)
            encoded_df=pd.DataFrame(results,columns=[f'{col}_MinHash' for col in cat_cols])
            combined_df=pd.concat([data,encoded_df],axis=1)
            
        else:
            results = encoder.fit_transform(data[cat_cols])
            # print(results)
            encoded_df=pd.DataFrame(results.toarray(),columns=encoder.get_feature_names_out())
            combined_df=pd.concat([data,encoded_df],axis=1)
     
        combined_df = combined_df.drop(cat_cols,axis=1)
        
        with open("encoder.pkl","wb") as enc:
            import pickle
            pickle.dump(encoder,enc)

        return combined_df
    