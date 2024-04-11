import pandas as pd
from sklearn.impute import SimpleImputer

class data_preprocessing:
    def __init__(self, data_path):
        self.df = pd.read_excel(data_path)

    def preprocess_data(self):
        # Missing values Treatment
        imputer = SimpleImputer(strategy='mean')
        self.df[['meduc', 'npvis', 'feduc']] = imputer.fit_transform(self.df[['meduc', 'npvis', 'feduc']])

        # Smoking binary feature
        self.df['smoking'] = self.df['cigs'].apply(lambda x: 0 if x == 0 else 1)

        # kilograms birth weight
        self.df['bwght_kg'] = (self.df['bwght'] / 1000).round(2)

        # outliers treatment
        self.df.loc[self.df['mage'] > 55, 'mage'] = 55

        return self.df






