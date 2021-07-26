import pandas as pd
import numpy as np
import glob
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed

DEFAULT_KWARGS = {"init": "random", "n_init": 50, "max_iter": 300,
                                 "random_state": 42}


def _get_inertia(df, n_clusters, kmeans_kwargs):
    kmeans = KMeans(n_clusters, **kmeans_kwargs)
    kmeans.fit(df)
    return kmeans.inertia_

def _get_coefficient(df, n_clusters, kmeans_kwargs):
    kmeans = KMeans(n_clusters, **kmeans_kwargs)
    kmeans.fit(df)
    return silhouette_score(df, kmeans.labels_)

class ClusteringUtility:
    """Contains clustering functionality"""
    

    @staticmethod
    def get_silhouette_for_plt_kmeans(df: np.ndarray,
                                          kmeans_kwargs=None, min_range: int = 2,
                                          max_range: int = 52, n_jobs: int = 8):
            """The silhouette value is a measure of how similar an object is to its own cluster
            (cohesion) compared to other clusters (separation). Expects a numpy.ndarray  that has
            had all data cleaned and standardized for Kmeans. Returns an array of silhouette_coefficients
            from each iteration of Kmeans so they can be plotted for best K clusters. Expects a dictionary
            of parameters for Kmeans, plus the min and max range of potential K clusters you could have.
            Note that if you want 50 clusters, you would need to start at 2 and expand to 52 as this is
            a range function and is exclusive. Notice you start with 2 clusters as it compares to how
            close the point is to the current cluster versus another cluster."""

            if kmeans_kwargs is None:
                kmeans_kwargs = DEFAULT_KWARGS

            if not isinstance(df, np.ndarray): 
                raise ValueError('need numpy.ndarray!')

            # Notice you start at 2 clusters for silhouette coefficient
            silhouette_coefficients = Parallel(n_jobs)(delayed(_get_coefficient)(df, k, kmeans_kwargs) for k in trange(min_range, max_range))
            
            return silhouette_coefficients

    
    @staticmethod
    def get_sse_for_elbow_plt_kmeans(df: np.ndarray,
                                     kmeans_kwargs=None, min_range: int = 1,
                                     max_range: int = 51, n_jobs: int = 8):
        """This calculates the Sum of Squared errors vs number of clusters and is used for finding the “elbow point”.
        The elbow point is where there is diminishing returns and is no longer worth the additional cost.
        In clustering, this means choosing a number of clusters so that adding another cluster doesn't
        give much better modeling of the data. Expects a numpy.ndarray that has had all data cleaned
        and standardized for Kmeans. Returns an array of sse from each iteration of Kmeans so
        they can be plotted for best K clusters. Expects a dictionary of parameters for Kmeans,
        plus the min and max range of potential K clusters you could have. Note that if you want 50
        clusters, you would need to start at 1, and expand to 51 as the range function is exclusive."""
 
        if kmeans_kwargs is None:
             kmeans_kwargs = DEFAULT_KWARGS

        if not isinstance(df, np.ndarray):
            raise ValueError('need numpy.ndarray!')
 
        sse = Parallel(n_jobs)(delayed(_get_inertia)(df, k, kmeans_kwargs) for k in trange(min_range, max_range))
        
        return sse


    @staticmethod    
    def fit_kmeans(df: np.ndarray, n_clusters: int = 50, kmeans_kwargs=None):
        """Expects a numpy.ndarray that has had all data cleaned and standardized for Kmeans.
        Returns the assigned clusters to each data point. Expects a K number of clusters and
        a dictionary of parameters for Kmeans."""

        if kmeans_kwargs is None:
            #kmeans_kwargs = {"init": "random", "n_init": 50, "max_iter": 300, "random_state": 42}
            kmeans_kwargs = DEFAULT_KWARGS
            

        if not isinstance(df, np.ndarray):
            raise ValueError('need numpy.ndarray!')

        kmeans = KMeans(n_clusters, **kmeans_kwargs)
        clusters = kmeans.fit_predict(df)
        return clusters