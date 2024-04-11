from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
import shap

class rd_model:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_random_forest_regressor(self):
        rf = RandomForestRegressor()
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 400]
        }
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.X, self.y)
        self.best_rf = grid_search.best_estimator_

    def evaluate_model(self, X_test, y_test):
        y_pred = self.best_rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2