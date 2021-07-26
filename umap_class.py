import pandas as pd
import numpy as np
import umap.umap_ as umap
from sklearn.preprocessing import PowerTransformer
 
    
class CreateUmapEmbeddings():
    
    '''Works for Kprototypes. Handles the catagorica/strings'''
    
    def __init__(self, df):
        self.df_ = df
        self.embedding_ = []
    
    # Process all int and float data types - uses a power transformation
    def preprocessing_Numerical(self):
        numerical = self.df_.select_dtypes(exclude=['object', 'category'])
 
        for c in numerical.columns:
            pt = PowerTransformer()
            numerical.loc[:, c] = pt.fit_transform(np.array(numerical[c]).reshape(-1, 1))
        return numerical
    
    # Process all the object data to dummied variables
    def preprocessing_Categorical(self):
        categorical = self.df_.select_dtypes(include=['category', 'object'])
        categorical = pd.get_dummies(categorical)
        return categorical 
    
    # Fit umap on numerical and dummied out object data
    def fit_umap_embeddings(self, numerical, categorical):
        #Percentage of columns which are categorical is used as weight parameter in embeddings later
        categorical_weight = len(self.df_.select_dtypes(include=['category', 'object']).columns) / self.df_.shape[1]
 
        #Embedding numerical & categorical
        fit1 = umap.UMAP(metric='l2').fit(numerical)
        fit2 = umap.UMAP(metric='dice').fit(categorical)
        return categorical_weight, fit1, fit2
    
    # Get final umap embeddings 
    def get_umap_embedding(self):
        
        numerical = self.preprocessing_Numerical()
        categorical = self.preprocessing_Categorical()
        categorical_weight, fit1, fit2 = self.fit_umap_embeddings(numerical, categorical)
        
        #Augmenting the numerical embedding with categorical
        intersection = umap.general_simplicial_set_intersection(fit1.graph_, fit2.graph_, weight=categorical_weight)
        intersection = umap.reset_local_connectivity(intersection)
        embedding = umap.simplicial_set_embedding(fit1._raw_data, intersection, fit1.n_components, 
                                                         fit1._initial_alpha, fit1._a, fit1._b, 
                                                         fit1.repulsion_strength, fit1.negative_sample_rate, 
                                                         200, 'random', np.random, fit1.metric, 
                                                         fit1._metric_kwds, False, {}, False)
        self.embedding_ = embedding[0]
        
        return embedding