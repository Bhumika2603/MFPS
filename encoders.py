from abc import ABC, abstractmethod
from glob import glob

import os
import numpy as np
import pandas as pd

from collections import defaultdict

from sklearn.preprocessing import OneHotEncoder
from category_encoders import count, cat_boost
from sklearn.decomposition import PCA

import pickle

from tqdm import tqdm
from typing import Callable, DefaultDict, Dict, Iterable




# %% ../nbs/11_Encoders.ipynb 8
class BaseEncoder(ABC):
    """
    An abstract base class to be used for all encoding transformations.

    Attributes
    ----------
    ABC : abc.ABC
        Abstract base class from the abc library.

    Methods
    -------
    __init__() -> None:
        Abstract initialization method.
    fit(*args, **kwargs):
        Abstract method to fit data.
    transform(*args, **kwargs):
        Abstract method to transform data.
    fit_transform(*args, **kwargs):
        Abstract method to fit and transform data.
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Abstract initialization method. This method is expected to be overridden by subclasses.

        Raises
        ------
        NotImplementedError
            If not implemented in a subclass.
        """
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Abstract method to fit data. This method is expected to be overridden by subclasses.

        Raises
        ------
        NotImplementedError
            If not implemented in a subclass.
        """
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        """
        Abstract method to transform data. This method is expected to be overridden by subclasses.

        Raises
        ------
        NotImplementedError
            If not implemented in a subclass.
        """
        pass

    @abstractmethod
    def fit_transform(self, *args, **kwargs):
        """
        Abstract method to fit and transform data. This method is expected to be overridden by subclasses.

        Raises
        ------
        NotImplementedError
            If not implemented in a subclass.
        """
        pass



# %% ../nbs/11_Encoders.ipynb 9
class DataEncoder(BaseEncoder):
    """Encoder that calls on multiple other encoders to do encoding flexibly"""

    def __init__(self,
                 enc_config: dict,  # Dictionary with all the relevant arguments to encode data
                 processed_columns: list,  # List of columns that are artificially created and processed already
                 ) -> None:
        self.enc_config = enc_config
        self.processed_columns = processed_columns
        print("processed coln are",self.processed_columns)

        self.encoders = defaultdict(dict)

    def fit(self, datax: pd.DataFrame, datay: pd.DataFrame = None):
        """Fit the encoders with the given data

        Parameters
        ----------
        datax : pd.DataFrame
            Data to fit the encoder on
        datay : pd.DataFrame, optional
            Labels for the given data, by default None
        """
        self.check_if_all_features_are_encoded(all_columns=set(datax.columns) - set(self.processed_columns))
        for encoding in self.encoders:
            if encoding == 'NoneEncoding':
                self.encoders[encoding]['encoder'] = NoneEncoder(self.encoders[encoding]['columns_to_encode'])
                self.encoders[encoding]['encoder'].fit(datax)

            elif encoding == 'OneHotEncoding':
                self.encoders[encoding]['encoder'] = OHE(self.encoders[encoding]['columns_to_encode'])
                self.encoders[encoding]['encoder'].fit(datax)

            elif encoding == 'NormalizedCountEncoding':
                # self.encoders[encoding]['columns_to_encode']+self.processed_columns
                self.encoders[encoding]['encoder'] = NormalizedCountEncoder(
                    self.encoders[encoding]['columns_to_encode'])
                self.encoders[encoding]['encoder'].fit(datax)

            elif encoding == 'CatBoostEncoding':
                self.encoders[encoding]['encoder'] = CatBoostEncoder(self.encoders[encoding]['columns_to_encode'])
                # print("datay is ",datay)
                self.encoders[encoding]['encoder'].fit(datax, datay)

            elif encoding == 'SimilarityEncoding':
                self.encoders[encoding]['encoder'] = SimilarityEncoder(self.encoders[encoding]['columns_to_encode'])
                self.encoders[encoding]['encoder'].fit(datax, datay)

            # elif encoding == 'BERTEncoding':
            #     self.encoders[encoding]['encoder'] = BERTEncoder(self.encoders[encoding]['columns_to_encode'])
            #     self.encoders[encoding]['encoder'].fit(datax)
                
            elif encoding == "GapEncoding":
                self.encoders[encoding]['encoder'] = Gap_Encoder(self.encoders[encoding]['columns_to_encode'])
                self.encoders[encoding]['encoder'].fit(datax)

            elif encoding == "MinHashEncoding":
                self.encoders[encoding]['encoder'] = MinHash_Encoder(self.encoders[encoding]['columns_to_encode'])
                self.encoders[encoding]['encoder'].fit(datax)
                
            else:
                raise (Exception(encoding + "hasn't been implemented yet"))

    def transform(self, datax: pd.DataFrame):
        """Transform the given data with the fitted encoders

        Parameters
        ----------
        datax : pd.DataFrame
            Data to encode

        Returns
        -------
        pd.DataFrame
            Encoded data
        """
        transformed_data = [datax[self.processed_columns]]
        for encoding in self.encoders:
            intermediate_encoded_data = self.encoders[encoding]['encoder'].transform(datax)
            transformed_data.append(intermediate_encoded_data)
        encoded_data = pd.concat(transformed_data, axis=1)
        encoded_data.index = datax.index
        encoded_data = encoded_data.sort_index(axis=1)
        return encoded_data

    def fit_transform(self, datax: pd.DataFrame, datay: pd.DataFrame = None) -> pd.DataFrame:
        """Fit the encoders with the given data and transform it

        Parameters
        ----------
        datax : pd.DataFrame
            Data to fit and transform
        datay : pd.DataFrame, optional
            Labels for the coresponding data, by default None

        Returns
        -------
        pd.DataFrame
            Encoded data
        """
        self.fit(datax, datay)
        return self.transform(datax)

    def check_if_all_features_are_encoded(self, all_columns: list):
        """Check if all the features of the raw data are covered among the different encoders

        Parameters
        ----------
        all_columns : list
            List of all the features in the raw data
        """
        columns_covered = set()
        for encoding in self.enc_config:
            if isinstance(self.enc_config[encoding], dict):
                columns_to_encode = sorted(list(set(all_columns) - set(self.enc_config[encoding]["all_except"])))
            elif isinstance(self.enc_config[encoding], list):
                columns_to_encode = sorted(self.enc_config[encoding])
            else:
                columns_to_encode = sorted(all_columns)
            self.encoders[encoding]['columns_to_encode'] = columns_to_encode
            columns_covered = columns_covered.union(columns_to_encode)
        leftover_columns = all_columns.difference(columns_covered)
        if len(leftover_columns) != 0:
            print("The following columns will be dropped as they haven't been assigned an encoding format: ", leftover_columns)


# %% ../nbs/11_Encoders.ipynb 11
class NoneEncoder(BaseEncoder):
    """
    This class is an encoder that renames the specified columns of a pandas DataFrame by appending '_None' to their names. 
    
    Parameters
    ----------
    columns_to_encode : list
        List of column names to be encoded (i.e., renamed).

    Attributes
    ----------
    final_columns : dict
        Dictionary that maps the original column names to the new column names.
    is_fit : bool
        Flag indicating whether the encoder has been fit.
    """

    def __init__(self, columns_to_encode: list) -> None:
        self.columns_to_encode = columns_to_encode
        self.final_columns = {column: column + '_None' for column in self.columns_to_encode}
        self.is_fit = False

    def fit(self, data: pd.DataFrame) -> None:
        """
        Sets the is_fit flag to True. No other operations are performed in this step.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the NoneEncoder. Note that this input is not used in the current implementation.

        Raises
        ------
        None
        """
        self.is_fit = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Renames the specified columns in the DataFrame, provided the encoder has been fit.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be transformed by the NoneEncoder.

        Returns
        -------
        pd.DataFrame
            Transformed data with the specified columns renamed.

        Raises
        ------
        Exception
            If the NoneEncoder has not been fit yet.
        """
        if not self.is_fit:
            raise Exception("NoneEncoder has not been fit yet")
        
        if data.shape[0] == 0:
            return pd.DataFrame(columns=self.columns_to_encode)

        encoded_data = data[self.columns_to_encode].copy()
        encoded_data.rename(self.final_columns, axis=1, inplace=True)
        return encoded_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the NoneEncoder and then transforms the given data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the NoneEncoder and transform it subsequently.

        Returns
        -------
        pd.DataFrame
            Transformed data with the specified columns renamed.

        Raises
        ------
        Exception
            If the NoneEncoder has not been fit yet.
        """
        self.fit(data)
        return self.transform(data)



# %% ../nbs/11_Encoders.ipynb 13
class OHE(BaseEncoder):
    """
    This class is an encoder that applies One-Hot Encoding (OHE) to the specified columns of a pandas DataFrame.
    
    Parameters
    ----------
    columns_to_encode : list
        List of column names to be encoded.

    Attributes
    ----------
    encoder : OneHotEncoder
        OneHotEncoder instance from sklearn.preprocessing.
    final_columns : list
        List of final column names after encoding.
    missing_columns : list
        List of column names with missing values after encoding.
    is_fit : bool
        Flag indicating whether the encoder has been fit.
    """

    def __init__(self, columns_to_encode: list) -> None:
        self.columns_to_encode = columns_to_encode
        self.encoder = OneHotEncoder(handle_unknown='ignore', dtype='uint8')
        self.final_columns = []
        self.missing_columns = []
        self.is_fit = False

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fits the OneHotEncoder on the specified columns and sets the is_fit flag to True.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the OneHotEncoder.

        Raises
        ------
        None
        """
        data_to_ohe = data[self.columns_to_encode].copy()
        self.encoder.fit(data_to_ohe)
        for i, feat in enumerate(self.columns_to_encode):
            for cats in self.encoder.categories_[i]:
                self.final_columns.append(feat + '__' + str(cats) + '_OneHot')
                if cats == 'missing_value':
                    self.missing_columns.append(feat + '__' + str(cats) + '_OneHot')
        self.is_fit = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame using the fitted OneHotEncoder, provided the encoder has been fit.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be transformed by the OneHotEncoder.

        Returns
        -------
        pd.DataFrame
            One-hot encoded data.

        Raises
        ------
        Exception
            If the OneHotEncoder has not been fit yet.
        """
        if not self.is_fit:
            raise Exception("OneHotEncoder has not been fit yet")
        
        if data.shape[0] == 0:
            return pd.DataFrame(columns=self.final_columns)
        
        data_to_ohe = data[self.columns_to_encode].copy()
        data_ohe = self.encoder.transform(data_to_ohe)
        encoded_data = pd.DataFrame.sparse.from_spmatrix(
            data_ohe, index=data.index, columns=self.final_columns).sparse.to_dense()
        encoded_data = encoded_data.drop(labels=self.missing_columns, axis=1, inplace=False)
        return encoded_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the OneHotEncoder and then transforms the given data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the OneHotEncoder and transform it subsequently.

        Returns
        -------
        pd.DataFrame
            One-hot encoded data.

        Raises
        ------
        Exception
            If the OneHotEncoder has not been fit yet.
        """
        self.fit(data)
        return self.transform(data)



# %% ../nbs/11_Encoders.ipynb 15
class NormalizedCountEncoder(BaseEncoder):
    """
    This class is an encoder that applies normalized count encoding to the specified columns of a pandas DataFrame. 
    Normalized count encoding replaces each categorical value with its frequency proportion in the dataset.
    
    Parameters
    ----------
    columns_to_encode : list
        List of column names to be encoded.

    Attributes
    ----------
    encoder : count.CountEncoder
        CountEncoder instance from the category_encoders library, set to normalize the counts.
    final_columns : dict
        Dictionary that maps the original column names to the new column names (original name + '_NormalizedCount').
    is_fit : bool
        Flag indicating whether the encoder has been fit.
    """

    def __init__(self, columns_to_encode: list) -> None:
        self.columns_to_encode = columns_to_encode
        self.encoder = count.CountEncoder(normalize=True)
        self.final_columns = {column: column + '_NormalizedCount' for column in self.columns_to_encode}
        self.is_fit = False

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fits the CountEncoder on the specified columns and sets the is_fit flag to True.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the NormalizedCountEncoder.

        Raises
        ------
        None
        """
        print(self.columns_to_encode)
        data_to_encode = data[self.columns_to_encode].copy()
        self.encoder.fit(data_to_encode)
        self.is_fit = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame using the fitted CountEncoder, provided the encoder has been fit.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be transformed by the NormalizedCountEncoder.

        Returns
        -------
        pd.DataFrame
            Normalized count encoded data.

        Raises
        ------
        Exception
            If the NormalizedCountEncoder has not been fit yet.
        """
        if not self.is_fit:
            raise Exception("NormalizedCountEncoder has not been fit yet")
        
        if data.shape[0] == 0:
            return pd.DataFrame(columns=self.final_columns)
        
        data_to_encode = data[self.columns_to_encode].copy()
        encoded_data = self.encoder.transform(data_to_encode)
        encoded_data.rename(self.final_columns, axis=1, inplace=True)
        encoded_data.index = data.index
        return encoded_data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the NormalizedCountEncoder and then transforms the given data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the NormalizedCountEncoder and transform it subsequently.

        Returns
        -------
        pd.DataFrame
            Normalized count encoded data.

        Raises
        ------
        Exception
            If the NormalizedCountEncoder has not been fit yet.
        """
        self.fit(data)
        return self.transform(data)



# %% ../nbs/11_Encoders.ipynb 17
class CatBoostEncoder(BaseEncoder):
    """
    This class is an encoder that applies CatBoost encoding to the specified columns of a pandas DataFrame. 
    CatBoost encoding is a target-based encoding method that uses gradient boosting concepts to reduce the target leakage.
    
    Parameters
    ----------
    columns_to_encode : list
        List of column names to be encoded.

    Attributes
    ----------
    encoder : cat_boost.CatBoostEncoder
        CatBoostEncoder instance from the category_encoders library.
    final_columns : list
        List of new column names after encoding (original name + '_CatBoost').
    is_fit : bool
        Flag indicating whether the encoder has been fit.
    """

    def __init__(self, columns_to_encode: list) -> None:
        self.columns_to_encode = columns_to_encode
        self.encoder = cat_boost.CatBoostEncoder()
        self.final_columns = [column + '_CatBoost' for column in self.columns_to_encode]
        self.is_fit = False

    def fit(self, data: pd.DataFrame, labels) -> None:
        """
        Fits the CatBoostEncoder on the specified columns and sets the is_fit flag to True.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the CatBoostEncoder.
        labels : array-like
            Target values.

        Raises
        ------
        None
        """
        data_to_encode = data[self.columns_to_encode].copy()
        self.encoder.fit(data_to_encode, labels)
        self.is_fit = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame using the fitted CatBoostEncoder, provided the encoder has been fit.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be transformed by the CatBoostEncoder.

        Returns
        -------
        pd.DataFrame
            CatBoost encoded data.

        Raises
        ------
        Exception
            If the CatBoostEncoder has not been fit yet.
        """
        if not self.is_fit:
            raise Exception("CatBoostEncoder has not been fit yet")
        
        if data.shape[0] == 0:
            return pd.DataFrame(columns=self.final_columns)
        
        data_to_encode = data[self.columns_to_encode].copy()
        encoded_data = self.encoder.transform(data_to_encode)
        encoded_data.columns = self.final_columns
        encoded_data.index = data.index
        return encoded_data

    def fit_transform(self, data: pd.DataFrame, labels) -> pd.DataFrame:
        """
        Fits the CatBoostEncoder and then transforms the given data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the CatBoostEncoder and transform it subsequently.
        labels : array-like
            Target values.

        Returns
        -------
        pd.DataFrame
            CatBoost encoded data.

        Raises
        ------
        Exception
            If the CatBoostEncoder has not been fit yet.
        """
        self.fit(data, labels)
        return self.transform(data)



# %% ../nbs/11_Encoders.ipynb 19
class MinHash_Encoder(BaseEncoder):

    def __init__(self, columns_to_encode: list) -> None:
        print("min hash ENcoder")
        self.columns_to_encode = columns_to_encode
        self.encoder = dirty_cat.MinHashEncoder(n_components=5)
        self.final_columns = []
        self.is_fit = False

    def fit(self, data: pd.DataFrame) -> None:
        data_to_encode = data[self.columns_to_encode].copy()
        self.encoder.fit(data_to_encode)

        for i, column in enumerate(self.columns_to_encode):
            for j in range(self.encoder.n_components):
                self.final_columns.append(column + '__' + str(j) + 'MinHash')
        self.is_fit = True

    def transform(self,  data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fit:
            raise Exception("MinHash has not been fit yet")
        
        if data.shape[0] == 0:
            return pd.DataFrame(columns=self.final_columns)
        
        data_to_encode = data[self.columns_to_encode].copy()
        encoded_data = pd.DataFrame.from_records(self.encoder.transform(data_to_encode))
        encoded_data.columns = self.final_columns
        encoded_data.index = data.index
        return encoded_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)



    # def __init__(self, columns_to_encode: list) -> None:
    #     print("min hash ENcoder")
    #     self.columns_to_encode = columns_to_encode
    #     self.encoder = dirty_cat.MinHashEncoder(n_components=5)
    #     self.final_columns = []
    #     self.is_fit = False

    # def fit(self, data: pd.DataFrame) -> None:
    #     data_to_encode = data[self.columns_to_encode].copy()
    #     self.encoder.fit(data_to_encode)

    #     for i, column in enumerate(self.columns_to_encode):
    #         for j in range(self.encoder.n_components):
    #             self.final_columns.append(column + '__' + str(j) + 'MinHash')
    #     self.is_fit = True

    # def transform(self,  data: pd.DataFrame) -> pd.DataFrame:
    #     if not self.is_fit:
    #         raise Exception("MinHash has not been fit yet")
        
    #     if data.shape[0] == 0:
    #         return pd.DataFrame(columns=self.final_columns)
    #     print(self.columns_to_encode)
    #     data_to_encode = data[self.columns_to_encode].copy()
    #     print("data_to_encode",data_to_encode.shape)
    #     data_to_encode.to_csv('data_to_encode_hash.csv')
    #     # print(self.encoder.transform(data_to_encode))
    #     # exit()
    #     encoded_data = pd.DataFrame.from_records(self.encoder.transform(data_to_encode))
    #     encoded_data.columns = self.final_columns
    #     encoded_data.index = data.index
    #     return encoded_data
    
    # def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
    #     self.fit(data)
    #     return self.transform(data)

# %% ../nbs/11_Encoders.ipynb 20
class Gap_Encoder(BaseEncoder):

    def __init__(self, columns_to_encode: list) -> None:
        print("GAP ENcoder")
        self.columns_to_encode = columns_to_encode
        self.encoder = dirty_cat.GapEncoder(n_components=5)
        self.final_columns = []
        self.is_fit = False

    def fit(self, data: pd.DataFrame) -> None:
        # print("to enc ",self.columns_to_encode,data.shape)

        data_to_encode = data[self.columns_to_encode].copy()
        data_to_encode = data_to_encode.astype(str)

        # data_to_encode.to_csv('data_to_encode.csv')
        # print(data_to_encode)
        self.encoder.fit(data_to_encode)

        for i, column in enumerate(self.columns_to_encode):
            for j in range(self.encoder.n_components):
                self.final_columns.append(column + '__' + str(j) + '_Gap')
        self.is_fit = True

    def transform(self,  data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fit:
            raise Exception("GapEncoder has not been fit yet")
        
        if data.shape[0] == 0:
            return pd.DataFrame(columns=self.final_columns)
        
        data_to_encode = data[self.columns_to_encode].copy()
        data_to_encode = data_to_encode.astype(str)

        encoded_data = pd.DataFrame.from_records(self.encoder.transform(data_to_encode))
        encoded_data.columns = self.final_columns
        encoded_data.index = data.index
        return encoded_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.astype(str)
        self.fit(data)
        print("Fit")
        return self.transform(data)

# %% ../nbs/11_Encoders.ipynb 21
class SimilarityEncoder(BaseEncoder):
    """
    This class is an encoder that applies Similarity encoding to the specified columns of a pandas DataFrame.
    Similarity encoding is a technique that encodes categorical variables based on a measure of how each category is 
    similar to each other category.
    
    Parameters
    ----------
    columns_to_encode : list
        List of column names to be encoded.

    Attributes
    ----------
    encoder : dirty_cat.SimilarityEncoder
        SimilarityEncoder instance from the dirty_cat library.
    final_columns : list
        List of new column names after encoding (original name + '__' + str(index) + '_Similarity').
    is_fit : bool
        Flag indicating whether the encoder has been fit.
    """

    def __init__(self, columns_to_encode: list) -> None:
        self.columns_to_encode = columns_to_encode
        self.encoder = dirty_cat.SimilarityEncoder(hashing_dim=5, categories='most_frequent', n_prototypes=10)
        self.final_columns = []
        self.is_fit = False

    def fit(self, data: pd.DataFrame, labels) -> None:
        """
        Fits the SimilarityEncoder on the specified columns and sets the is_fit flag to True.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the SimilarityEncoder.
        labels : array-like
            Target values.

        Raises
        ------
        None
        """
        data_to_encode = data[self.columns_to_encode].copy()
        self.encoder.fit(data_to_encode, labels)
        repeat_list = [len(x) for x in self.encoder.categories_]
        for i, column in enumerate(self.columns_to_encode):
            for j in range(repeat_list[i]):
                self.final_columns.append(column + '__' + str(j) + '_Similarity')
        self.is_fit = True

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame using the fitted SimilarityEncoder, provided the encoder has been fit.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be transformed by the SimilarityEncoder.

        Returns
        -------
        pd.DataFrame
            Similarity encoded data.

        Raises
        ------
        Exception
            If the SimilarityEncoder has not been fit yet.
        """
        if not self.is_fit:
            raise Exception("SimilarityEncoder has not been fit yet")
        
        if data.shape[0] == 0:
            return pd.DataFrame(columns=self.final_columns)
        
        data_to_encode = data[self.columns_to_encode].copy()
        encoded_data = pd.DataFrame.from_records(self.encoder.transform(data_to_encode))
        encoded_data.columns = self.final_columns
        encoded_data.index = data.index
        return encoded_data

    def fit_transform(self, data: pd.DataFrame, labels) -> pd.DataFrame:
        """
        Fits the SimilarityEncoder and then transforms the given data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to fit the SimilarityEncoder and transform it subsequently.
        labels : array-like
            Target values.

        Returns
        -------
        pd.DataFrame
            Similarity encoded data.

        Raises
        ------
        Exception
            If the SimilarityEncoder has not been fit yet.
        """
        self.fit(data, labels)
        return self.transform(data)


# %% ../nbs/11_Encoders.ipynb 23
# class BERTEncoder():
#     def __init__(self, encode_covariates: Iterable):
#         self.pca: Dict[str, PCA] = dict()
#         self.encode_covariates = encode_covariates
#         self.DEFAULT_STR = 'DEFAULT_STR'
#         self.is_fit = False
#         self.final_columns = []

#     def create_bert(self):
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.bert_instance = BertModel.from_pretrained('bert-base-uncased').to("cuda:0")

#     def fit(self, data: pd.DataFrame) -> None:
#         self.create_bert()

        
#         for column in tqdm(self.encode_covariates,  desc="fitting"):
#             categories = data[column].replace(np.nan, self.DEFAULT_STR).unique().tolist()
#             embeddings = []
#             for category in categories:
#                 input_ids = self.tokenizer(str(category), return_tensors="pt").to("cuda:0")["input_ids"]
#                 embeddings.append(
#                     self.bert_instance(input_ids)[0].squeeze(0).mean(dim=0).cpu().detach().numpy()
#                 )
#             embeddings = np.array(embeddings)
#             _pca = PCA(n_components=min(12, *embeddings.shape))
#             _pca.fit(embeddings)
#             self.pca[column] = _pca

#             self.final_columns += [f"{column}_{i}_BERTEncoding" for i in range(min(12, *embeddings.shape))]

#         self.is_fit = True

#     def get_embedding(self, word: str, pca_instance: PCA):
#         input_ids = self.tokenizer(word, return_tensors="pt").to("cuda:0")["input_ids"]  # Batch size 1
#         outputs = self.bert_instance(input_ids)
#         last_hidden_states = outputs[0].mean(dim=1).cpu().detach().numpy()
#         return pca_instance.transform(last_hidden_states)[0]

#     def transform(self, data: pd.DataFrame) -> pd.DataFrame:

#         if not self.is_fit:
#             raise Exception("BERTEncoder has not been fit yet")

#         result_dfs = list()

#         data = data[self.encode_covariates]

#         for column in tqdm(self.encode_covariates, desc="transforming"):
#             categories = data[column].copy().replace(np.nan, self.DEFAULT_STR)
#             unique_categories = categories.unique().tolist()
#             unique_category_embeddings = {
#                 category: self.get_embedding(category, self.pca[column])
#                 for category in tqdm(unique_categories)
#             }

#             result_dfs.append(
#                 pd.DataFrame.from_records(
#                     [
#                         unique_category_embeddings[category]
#                         for category in tqdm(categories.tolist(), desc="transforming one column", leave=False)
#                     ], columns=[f"{column}_{i}_BertEncoding" for i in range(len(list(unique_category_embeddings.values())[0]))]
#                 )
#             )

#         # https://stackoverflow.com/questions/38978214/merge-a-list-of-dataframes-to-create-one-dataframe
#         dfs = [df.reset_index(drop=True) for df in result_dfs]
#         trainx_encoded = pd.concat(dfs, axis=1)
    
#         trainx_encoded.index = data.index
#         trainx_transformed = data.drop(self.encode_covariates, axis=1).merge(
#             trainx_encoded,
#             left_index=True,
#             right_index=True
#         )
        
#         return trainx_transformed
    
#     def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
#         self.fit(data)
#         return self.transform(data)