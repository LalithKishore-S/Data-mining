import pandas as pd
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.cpt = None

    def smoothing(self, k = 3):
        for feature in self.cpt.keys():
            if feature in self.numerical_features:
                continue
            for category in self.cpt[feature].keys():
                for label in self.classes:
                    if self.cpt[feature][category][label] == 0:
                        for category1 in self.cpt[feature].keys():
                            self.cpt[feature][category1][label] += k
            
            for label in self.classes:
                tot_counts = np.sum([self.cpt[feature][category][label] for category in self.cpt[feature].keys()])
                for cat in self.cpt[feature].keys():
                    self.cpt[feature][cat][label] /= tot_counts

    def fit(self, X, y):

        self.features = X.columns
        self.prior_proba = {label : y[y==label].shape[0]/y.shape[0] for label in np.unique(y)}
        self.categorical_features = X.select_dtypes('object').columns
        self.numerical_features = X.select_dtypes(['int64', 'float64']).columns
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.cpt = {}
        y = np.array(y)

        for feature in self.features:
            self.cpt[feature] = {}
            if feature in self.numerical_features:
                for label in self.classes:
                    self.cpt[feature][label] = {}
                    index = np.where(y == label)[0]
                    data = X.iloc[index,:][feature]
                    self.cpt[feature][label]["mean"] = data.mean()
                    self.cpt[feature][label]["std"] = data.std()

            else:
                for category in X[feature].unique():
                    self.cpt[feature][category] = {}
                    for label in self.classes:
                        self.cpt[feature][category][label] = X[(X[feature] == category) & (y == label)].shape[0]

        self.smoothing(k = 3)

    def predict_proba(self,test_data):
        pred = {}
        denom = 0
        for label in self.classes:
            numerator = self.prior_proba[label]
            for feature in test_data.keys():
                if feature in self.categorical_features:
                    numerator *= self.cpt[feature][test_data[feature]][label]
                else:
                    mean = self.cpt[feature][label]["mean"]
                    std = self.cpt[feature][label]["std"]
                    numerator *= 1 / (std * (2 * np.pi) ** 0.5) * np.exp(-(test_data[feature] - mean) ** 2/ (2 * std ** 2) )

            pred[label] = numerator
            denom += numerator

        pred = {label: val/denom for label, val in pred.items()}
        return pred
    

data = {
    'Age':        [25, 45, 35, 50, 23, 40, 60, 33, 47, 29, 38, 55, 42, 31, 48],
    'Income':     [30000, 80000, 50000, 120000, 25000, 70000, 95000, 45000, 85000, 32000, 60000, 110000, 75000, 40000, 90000],
    'Employment': ['Salaried','Self-Employed','Salaried','Salaried','Unemployed',
                   'Salaried','Self-Employed','Salaried','Salaried','Unemployed',
                   'Self-Employed','Salaried','Salaried','Salaried','Self-Employed'],
    'Education':  ['Graduate','Post-Graduate','Graduate','Post-Graduate','UnderGrad',
                   'Graduate','Post-Graduate','Graduate','Post-Graduate','UnderGrad',
                   'Graduate','Post-Graduate','Graduate','UnderGrad','Post-Graduate'],
    'Credit_Score': [650, 750, 700, 800, 580, 720, 780, 690, 760, 600, 710, 790, 740, 670, 770],
    'Loan_Amount':  [10000, 50000, 25000, 80000, 8000, 40000, 60000, 20000, 55000, 12000, 35000, 75000, 45000, 18000, 65000],
    'Approved':     [0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop('Approved', axis=1)
y = df['Approved']

nb = NaiveBayes()
nb.fit(X, y)
print(nb.predict_proba(X.iloc[0].to_dict()))

