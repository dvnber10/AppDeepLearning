import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Cargar datos
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')  # Descarga test.csv desde Kaggle

# Combinar train y test para preprocesamiento
all_data = pd.concat([train, test], ignore_index=False)

# Manejo de valores faltantes
none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
for col in none_cols:
    all_data[col] = all_data[col].fillna('None')

zero_cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
             'GarageYrBlt', 'GarageCars', 'GarageArea']
for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)

# Imputar LotFrontage por mediana del vecindario
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# Imputar otros valores faltantes
remaining_cols = all_data.columns[all_data.isnull().any()].drop('SalePrice', errors='ignore')
for col in remaining_cols:
    if all_data[col].dtype == 'object':
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    else:
        all_data[col] = all_data[col].fillna(all_data[col].median())

# Ingeniería de características avanzada
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch']
all_data['OverallQual_SF'] = all_data['OverallQual'] * all_data['TotalSF']  # Interacción calidad y área
all_data['LotArea_per_Room'] = all_data['LotArea'] / (all_data['TotRmsAbvGrd'] + 1)  # Área por habitación

# Eliminar outliers en train
train = all_data[all_data['SalePrice'].notna()]
outliers = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index
train = train.drop(outliers)

# Transformación logarítmica de características sesgadas
numerical_feats = train.select_dtypes(include=['int64', 'float64']).columns.drop(['SalePrice', 'Id'], errors='ignore')
skewed_feats = train[numerical_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]
for feat in skewed_feats.index:
    all_data[feat] = np.log1p(all_data[feat])

# Mapeo de características ordinales
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

# Eliminar columnas con baja varianza o redundantes
all_data = all_data.drop(['Utilities', 'Street', 'PoolQC', 'YrSold', 'MoSold'], axis=1)

# Dividir nuevamente en train y test
train = all_data[all_data['SalePrice'].notna()]
test = all_data[all_data['SalePrice'].isna()]

# Transformar el target (SalePrice)
y = np.log1p(train['SalePrice'])
X = train.drop(['SalePrice', 'Id'], axis=1)

# Columnas categóricas y numéricas
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Pipeline de preprocesamiento
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

# Definir modelos base para stacking
estimators = [
    ('xgb', XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=0)),
    ('lgb', LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)),
    ('cat', CatBoostRegressor(n_estimators=1000, learning_rate=0.05, depth=3, subsample=0.8, verbose=0, random_state=0)),
    ('en', ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=0))
]

# Stacking con un modelo final
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0),
    cv=5
)

# Pipeline completo
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', stack)])

# Validación cruzada para evaluar el modelo
kf = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(my_pipeline, X, y, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"Cross-validation RMSLE scores: {rmse_scores}")
print(f"Mean RMSLE: {rmse_scores.mean():.6f} (+/- {rmse_scores.std() * 2:.6f})")

# Entrenar el modelo
my_pipeline.fit(X, y)

# Predecir en test
test_X = test.drop(['SalePrice', 'Id'], axis=1)
preds_log = my_pipeline.predict(test_X)
preds = np.expm1(preds_log)

# Crear archivo de submission
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds})
submission.to_csv('submission2.csv', index=False)
print("Archivo de submission creado: submission.csv")