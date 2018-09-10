import numpy as np
import shogun as sg
from sklearn.base import BaseEstimator, TransformerMixin

class TapkeeProjection(BaseEstimator, TransformerMixin):
    def __init__(self, converter):
        self.converter = converter
        self.converter.parallel.set_num_threads(8)

    def fit_transform(self, X, y=None):
        features = sg.RealFeatures(X.T.astype('float64'))
        return self.converter.embed(features).get_feature_matrix().T

class DiffusionMaps(TapkeeProjection):
    def __init__(self, n_components=2, t=2, width=10.0):
        super(DiffusionMaps, self).__init__(converter=sg.DiffusionMaps())
        self.set_params(n_components, t, width)

    def set_params(self, n_components=2, t=2, width=10.0):
        self.converter.set_target_dim(n_components)
        self.converter.set_t(t)
        self.converter.set_width(width)
        self.converter.set_distance(sg.EuclideanDistance())

class ManifoldSculpting(TapkeeProjection):
    def __init__(self, n_components=2, n_neighbors=10, squishing_rate=0.8, max_iter=80):
        super(ManifoldSculpting, self).__init__(converter=sg.ManifoldSculpting())
        self.set_params(n_components, n_neighbors, squishing_rate, max_iter)

    def set_params(self, n_components=2, n_neighbors=10, squishing_rate=0.8, max_iter=80):
        self.converter.set_target_dim(n_components)
        self.converter.set_k(n_neighbors)
        self.converter.set_squishing_rate(squishing_rate)
        self.converter.set_max_iteration(max_iter)
        self.converter.set_distance(sg.EuclideanDistance())

class StochasticProximityEmbedding(TapkeeProjection):
    def __init__(self, n_components=2, n_neighbors=12, n_updates=20, max_iter=0):
        super(StochasticProximityEmbedding, self).__init__(converter=sg.StochasticProximityEmbedding())
        self.set_params(n_components, n_neighbors, n_updates, max_iter)

    def set_params(self, n_components=2, n_neighbors=12, n_updates=20, max_iter=0):
        self.converter.set_target_dim(n_components)
        self.converter.set_k(n_neighbors)
        self.converter.set_nupdates(n_updates)
        self.converter.set_max_iteration(max_iter)
        self.converter.set_distance(sg.EuclideanDistance())

class HessianLocallyLinearEmbedding(TapkeeProjection):
    def __init__(self, n_components=2, n_neighbors=10):
        super(HessianLocallyLinearEmbedding, self).__init__(converter=sg.HessianLocallyLinearEmbedding())
        self.set_params(n_components, n_neighbors)

    def set_params(self, n_components=2, n_neighbors=10):
        self.converter.set_target_dim(n_components)
        self.converter.set_k(n_neighbors)
        self.converter.set_distance(sg.EuclideanDistance())
        kernel = sg.GaussianKernel(100, 10.0)
        self.converter.set_kernel(kernel)

class KernelLocallyLinearEmbedding(TapkeeProjection):
    def __init__(self, n_components=2, n_neighbors=10):
        super(KernelLocallyLinearEmbedding, self).__init__(converter=sg.KernelLocallyLinearEmbedding())
        self.set_params(n_components, n_neighbors)

    def set_params(self, n_components=2, n_neighbors=10):
        self.converter.set_target_dim(n_components)
        self.converter.set_k(n_neighbors)
        self.converter.set_distance(sg.EuclideanDistance())
        kernel = sg.GaussianKernel(100, 10.0)
        self.converter.set_kernel(kernel)


class LocalityPreservingProjections(TapkeeProjection):
    def __init__(self, n_components=2, n_neighbors=10):
        super(LocalityPreservingProjections, self).__init__(converter=sg.LocalityPreservingProjections())
        self.set_params(n_components, n_neighbors)

    def set_params(self, n_components=2, n_neighbors=10):
        self.converter.set_target_dim(n_components)
        self.converter.set_k(n_neighbors)
        self.converter.set_distance(sg.EuclideanDistance())

class LinearLocalTangentSpaceAlignment(TapkeeProjection):
    def __init__(self, n_components=2, n_neighbors=10):
        super(LinearLocalTangentSpaceAlignment, self).__init__(converter=sg.LinearLocalTangentSpaceAlignment())
        self.set_params(n_components, n_neighbors)

    def set_params(self, n_components=2, n_neighbors=10):
        self.converter.set_target_dim(n_components)
        self.converter.set_k(n_neighbors)
        self.converter.set_distance(sg.EuclideanDistance())

class NeighborhoodPreservingEmbedding(TapkeeProjection):
    def __init__(self, n_components=2, n_neighbors=10):
        super(NeighborhoodPreservingEmbedding, self).__init__(converter=sg.NeighborhoodPreservingEmbedding())
        self.set_params(n_components, n_neighbors)

    def set_params(self, n_components=2, n_neighbors=10):
        self.converter.set_target_dim(n_components)
        self.converter.set_k(n_neighbors)
        self.converter.set_distance(sg.EuclideanDistance())
