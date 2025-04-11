import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, SelectKBest
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso

# EDA
total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv", sep = ",")
total_data.to_csv("../data/raw/total_data.csv", index = False)

total_data = total_data.drop_duplicates().reset_index(drop = True)

## Factorización de columnas no numéricas
object_columns = total_data.select_dtypes(include='object').columns
total_data["COUNTY_NAME_N"] = pd.factorize(total_data["COUNTY_NAME"])[0]
total_data["STATE_NAME_N"] = pd.factorize(total_data["STATE_NAME"])[0]

columnas = [
    ("COUNTY_NAME_N", "COUNTY_NAME"),
    ("STATE_NAME_N", "STATE_NAME"),
]

transformation_rules = {}

for original_col, normalized_col in columnas:
    mapping = {
        row[original_col]: row[normalized_col]
        for _, row in total_data[[original_col, normalized_col]].drop_duplicates().iterrows()
    }
    transformation_rules[original_col] = mapping


with open("../models/transformation_rules.json", "w") as f:
    json.dump(transformation_rules, f, indent=4)

## Correlación de variables y eliminado de las que tienen alta correlación

target = "Heart disease_number"

numeric_df = total_data.select_dtypes(include=['number'])

corr_matrix = numeric_df.corr().abs()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
corr_matrix_masked = corr_matrix.where(~mask)

to_drop = set()

for col in corr_matrix_masked.columns:
    for row in corr_matrix_masked.index:
        if corr_matrix_masked.loc[row, col] > 0.9:
            if col != target and row != target:
                to_drop.add(col)

total_data_reduced = total_data.drop(columns=to_drop)

total_data_reduced.to_csv("../data/raw/total_data_reduced.csv", index = False)

## Feature selection
total_data_con_outliers = total_data_reduced.copy()
total_data_sin_outliers = total_data_reduced.copy()

def replace_outliers_from_column(column, df):
  column_stats = df[column].describe()
  column_iqr = column_stats["75%"] - column_stats["25%"]
  upper_limit = column_stats["75%"] + 1.5 * column_iqr
  lower_limit = column_stats["25%"] - 1.5 * column_iqr

  lower_limit = float(df[column].min())
  # Remove upper outliers
  df[column] = df[column].apply(lambda x: x if (x <= upper_limit) else upper_limit)
  # Remove lower outliers
  df[column] = df[column].apply(lambda x: x if (x >= lower_limit) else lower_limit)
  return df.copy(), [lower_limit, upper_limit]

outliers_dict = {}
for column in total_data_sin_outliers.select_dtypes(include=['number']).columns:
    total_data_sin_outliers, limits_list = replace_outliers_from_column(column, total_data_sin_outliers)
    outliers_dict[column] = limits_list

with open("../models/outliers_replacement.json", "w") as f:
    json.dump(outliers_dict, f)

## Escalado de valores
numeric_columns = total_data_reduced.select_dtypes(include=['number']).columns.drop(target)

X_con_outliers = total_data_con_outliers[numeric_columns]
X_sin_outliers = total_data_sin_outliers[numeric_columns]
y = total_data_con_outliers[target]

X_train_con_outliers, X_test_con_outliers, y_train, y_test = train_test_split(X_con_outliers, y, test_size = 0.2, random_state = 42)
X_train_sin_outliers, X_test_sin_outliers = train_test_split(X_sin_outliers, test_size = 0.2, random_state = 42)


X_train_con_outliers.to_excel("../data/processed/X_train_con_outliers.xlsx", index = False)
X_train_sin_outliers.to_excel("../data/processed/X_train_sin_outliers.xlsx", index = False)
X_test_con_outliers.to_excel("../data/processed/X_test_con_outliers.xlsx", index = False)
X_test_sin_outliers.to_excel("../data/processed/X_test_sin_outliers.xlsx", index = False)
y_train.to_excel("../data/processed/y_train.xlsx", index = False)
y_test.to_excel("../data/processed/y_test.xlsx", index = False)

### Normalizacion
normalizador_con_outliers = StandardScaler()
normalizador_con_outliers.fit(X_train_con_outliers)

with open("../models/normalizador_con_outliers.pkl", "wb") as file:
    pickle.dump(normalizador_con_outliers,file)

X_train_con_outliers_norm = normalizador_con_outliers.transform(X_train_con_outliers)
X_train_con_outliers_norm = pd.DataFrame(X_train_con_outliers_norm, index = X_train_con_outliers.index, columns = numeric_columns)

X_test_con_outliers_norm = normalizador_con_outliers.transform(X_test_con_outliers)
X_test_con_outliers_norm = pd.DataFrame(X_test_con_outliers_norm, index = X_test_con_outliers.index, columns = numeric_columns)

X_train_con_outliers_norm.to_excel("../data/processed/X_train_con_outliers_norm.xlsx", index = False)
X_test_con_outliers_norm.to_excel("../data/processed/X_test_con_outliers_norm.xlsx", index = False)

normalizador_sin_outliers = StandardScaler()
normalizador_sin_outliers.fit(X_train_sin_outliers)

with open("../models/normalizador_sin_outliers.pkl", "wb") as file:
    pickle.dump(normalizador_sin_outliers,file)

X_train_sin_outliers_norm = normalizador_sin_outliers.transform(X_train_sin_outliers)
X_train_sin_outliers_norm = pd.DataFrame(X_train_sin_outliers_norm, index = X_train_sin_outliers.index, columns = numeric_columns)

X_test_sin_outliers_norm = normalizador_sin_outliers.transform(X_test_sin_outliers)
X_test_sin_outliers_norm = pd.DataFrame(X_test_sin_outliers_norm, index = X_test_sin_outliers.index, columns = numeric_columns)

X_train_sin_outliers_norm.to_excel("../data/processed/X_train_sin_outliers_norm.xlsx", index = False)
X_test_sin_outliers_norm.to_excel("../data/processed/X_test_sin_outliers_norm.xlsx", index = False)

### Min-max
min_max_con_outliers = MinMaxScaler()
min_max_con_outliers.fit(X_train_con_outliers)

with open("../models/min_max_con_outliers.pkl", "wb") as file:
    pickle.dump(min_max_con_outliers,file)

X_train_con_outliers_scal = min_max_con_outliers.transform(X_train_con_outliers)
X_train_con_outliers_scal = pd.DataFrame(X_train_con_outliers_scal, index = X_train_con_outliers.index, columns = numeric_columns)

X_test_con_outliers_scal = min_max_con_outliers.transform(X_test_con_outliers)
X_test_con_outliers_scal = pd.DataFrame(X_test_con_outliers_scal, index = X_test_con_outliers.index, columns = numeric_columns)

X_train_con_outliers_scal.to_excel("../data/processed/X_train_con_outliers_scal.xlsx", index = False)
X_test_con_outliers_scal.to_excel("../data/processed/X_test_con_outliers_scal.xlsx", index = False)

min_max_sin_outliers = MinMaxScaler()
min_max_sin_outliers.fit(X_train_sin_outliers)

with open("../models/min_max_sin_outliers.pkl", "wb") as file:
    pickle.dump(min_max_sin_outliers,file)

X_train_sin_outliers_scal = min_max_sin_outliers.transform(X_train_sin_outliers)
X_train_sin_outliers_scal = pd.DataFrame(X_train_sin_outliers_scal, index = X_train_sin_outliers.index, columns = numeric_columns)

X_test_sin_outliers_scal = min_max_sin_outliers.transform(X_test_sin_outliers)
X_test_sin_outliers_scal = pd.DataFrame(X_test_sin_outliers_scal, index = X_test_sin_outliers.index, columns = numeric_columns)

X_train_sin_outliers_scal.to_excel("../data/processed/X_train_sin_outliers_scal.xlsx", index = False)
X_test_sin_outliers_scal.to_excel("../data/processed/X_test_sin_outliers_scal.xlsx", index = False)

## Feature selection
selection_model = SelectKBest(f_classif, k = 10)
selection_model.fit(X_train_con_outliers_scal, y_train)

ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train_con_outliers_scal), columns = X_train_con_outliers_scal.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test_con_outliers_scal), columns = X_test_con_outliers_scal.columns.values[ix])

with open("../models/feature_selection_k_10.json", "w") as f:
    json.dump(X_train_sel.columns.tolist(), f)

X_train_sel[target] = list(y_train)
X_test_sel[target] = list(y_test)

X_train_sel.to_csv("../data/processed/clean_train.csv", index=False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index=False)

# MACHINE LEARNING

BASE_PATH = "../data/processed"
TRAIN_PATHS = [
    "X_train_con_outliers.xlsx",
    "X_train_sin_outliers.xlsx",
    "X_train_con_outliers_norm.xlsx",
    "X_train_sin_outliers_norm.xlsx",
    "X_train_con_outliers_scal.xlsx",
    "X_train_sin_outliers_scal.xlsx"
]
TRAIN_DATASETS = []
for path in TRAIN_PATHS:
    TRAIN_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

TEST_PATHS = [
    "X_test_con_outliers.xlsx",
    "X_test_sin_outliers.xlsx",
    "X_test_con_outliers_norm.xlsx",
    "X_test_sin_outliers_norm.xlsx",
    "X_test_con_outliers_scal.xlsx",
    "X_test_sin_outliers_scal.xlsx"
]
TEST_DATASETS = []
for path in TEST_PATHS:
    TEST_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

y_train = pd.read_excel(f"{BASE_PATH}/y_train.xlsx")
y_test = pd.read_excel(f"{BASE_PATH}/y_test.xlsx")

## Regresión lineal
results = []
models=[]

for index, dataset in enumerate(TRAIN_DATASETS):
    model = LinearRegression()
    model.fit(dataset, y_train)
    models.append(model)
    
    y_pred_train = model.predict(dataset)
    y_pred_test = model.predict(TEST_DATASETS[index])

    results.append(
        {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "test_r2": r2_score(y_test, y_pred_test)
        }
    )

best_model=3
final_model = models[best_model]

with open("../models/linear_best_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("../models/final_results_linear.json", "w") as f:
    json.dump(results, f, indent=4)

## Regresión lineal regularizada
alphas = np.linspace(0.0, 20.0, 100)

lasso_models = []
lasso_results = []
train_r2_scores = []
test_r2_scores = []

X_train = TRAIN_DATASETS[4]
X_test = TEST_DATASETS[4]

alphas = np.linspace(0.0, 20.0, 100)
lasso_models = []
lasso_results = []
train_r2_scores = []
test_r2_scores = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    lasso_models.append(lasso)
    train_r2_scores.append(train_r2)
    test_r2_scores.append(test_r2)

    lasso_results.append({
        "alpha": alpha,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred)
    })

best_index_lasso = np.argmax(test_r2_scores)
best_model_lasso = lasso_models[best_index_lasso]

with open("../models/lasso_best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("../models/final_results_lasso.json", "w") as f:
    json.dump(lasso_results, f, indent=4)

### Gráfico de evolución de R2
plt.figure(figsize=(10, 6))
plt.plot(alphas, train_r2_scores, label='Train R²')
plt.plot(alphas, test_r2_scores, label='Test R²')
plt.xlabel('Alpha')
plt.ylabel('R² Score')
plt.title('Evolución del R² con Lasso')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()