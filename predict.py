from sklearn.model_selection import cross_val_score
import numpy as np

class prediction_new_data:
    def __init__(self, model):
        self.model = model

    def cross_validation(self):
        scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_squared_error')
        mse_scores = -scores
        mean_mse = np.mean(mse_scores)
        return mse_scores, mean_mse

    def predict(self, X_test):
        return self.model.predict(X_test)
