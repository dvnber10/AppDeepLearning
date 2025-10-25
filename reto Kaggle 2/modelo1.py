import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data - replace with your file paths
# For the provided train data, you can save the <DOCUMENT> content as 'train.csv'
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')  # Download test.csv from Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# Combine train and test for preprocessing
all_data = pd.concat([train, test], ignore_index=False)

# Handle missing values
# Features where NA means "none"
none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
for col in none_cols:
    all_data[col] = all_data[col].fillna('None')

# Features where NA means 0
zero_cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
             'GarageYrBlt', 'GarageCars', 'GarageArea']
for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)

# Impute median for LotFrontage based on neighborhood
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Other missing values - mode for categorical, median for numerical
remaining_cols = all_data.columns[all_data.isnull().any()].drop('SalePrice', errors='ignore')
for col in remaining_cols:
    if all_data[col].dtype == 'object':
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    else:
        all_data[col] = all_data[col].fillna(all_data[col].median())

# Feature engineering
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']

# Drop utilities as it's mostly one value
all_data = all_data.drop(['Utilities', 'Street', 'PoolQC', 'YrSold', 'MoSold'], axis=1)

# Log transform skewed numerical features
numerical_feats = all_data.select_dtypes(include=['int64', 'float64']).columns.drop('Id', errors='ignore')
skewed_feats = all_data[numerical_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]
for feat in skewed_feats.index:
    all_data[feat] = np.log1p(all_data[feat])

# Ordinal features mapping
ordinal_cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'ExterQual', 'ExterCond', 
                'HeatingQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'BsmtExposure', 
                'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'CentralAir', 'MSSubClass']
ordinal_mapping = {
    'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0,
    'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'Missing': 0,
    'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0,
    'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1,
    'Y': 1, 'N': 0,
    'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0,
    'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1,
    'Av': 3, 'Mn': 2, 'No': 1, 'None': 0,
    'Sev': 3, 'Mod': 2, 'Gtl': 1
}
for col in ordinal_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].map(ordinal_mapping).fillna(0)

# Split back to train and test
train = all_data[all_data['SalePrice'].notna()]
test = all_data[all_data['SalePrice'].isna()]

# Target log transform
y = np.log1p(train['SalePrice'])
X = train.drop(['SalePrice', 'Id'], axis=1)

# Categorical columns for one-hot encoding
categorical_cols = X.select_dtypes(include=['object']).columns

# Numerical columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Model - XGBoost with tuning
model = XGBRegressor(random_state=0)

# Pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

# Hyperparameter tuning
param_grid = {
    'model__n_estimators': [10000],
    'model__learning_rate': [0.005],
    'model__max_depth': [3],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8]
}

search = GridSearchCV(my_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
search.fit(X, y)

print("Best params:", search.best_params_)
print("Best score (RMSE):", np.sqrt(-search.best_score_))

# Predict on test
test_X = test.drop(['SalePrice', 'Id'], axis=1)
preds_log = search.predict(test_X)
preds = np.expm1(preds_log)

# Submission
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds})
submission.to_csv('submission.csv', index=False)
print("Archivo de submission creado: submission.csv")