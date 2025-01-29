from sklearn.base import BaseEstimator, TransformerMixin


# Create a temporary scaler that performs no scaling
class NoScalingScaler(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return x

    def inverse_transform(self, x, y=None):
        return x