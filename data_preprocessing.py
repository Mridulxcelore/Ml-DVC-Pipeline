# import pandas as pd
# from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# from sklearn.feature_selection import mutual_info_classif, RFE, SelectFromModel
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from category_encoders.target_encoder import TargetEncoder
#
# data = pd.read_csv("train.csv")
#
# shape = data.shape
# print(f"Shape: {shape}")
# print(f"Columns: {list(data.columns)}")
# print("\nNumber of null values:\n")
# print(data.isnull().sum())
# print("\nData information:\n")
# print(data.info())
#
# categorical_columns = [
#     "gender",
#     "ethnicity",
#     "education_level",
#     "income_level",
#     "smoking_status",
#     "employment_status"
# ]
#
# categorical_data = data[categorical_columns].copy()
# categorical_data.to_csv("categorical_data.csv", index=False)
#
# ordinal_cols = ["education_level", "income_level", "smoking_status"]
# onehot_cols = ["gender", "employment_status"]
# target_cols = ["ethnicity"]
#
# y = data["diagnosed_diabetes"]
#
# print("\nValues of education_level")
# print(categorical_data["education_level"].unique())
#
# print("\nValues of income_level")
# print(categorical_data["income_level"].unique())
#
# print("\nValues of smoking_status")
# print(categorical_data["smoking_status"].unique())
#
# education_order = [
#     "No formal",
#     "Highschool",
#     "Graduate",
#     "Postgraduate"
# ]
#
# income_order = [
#     "Low",
#     "Lower-Middle",
#     "Middle",
#     "Upper-Middle",
#     "High"
# ]
#
# smoking_order = [
#     "Never",
#     "Former",
#     "Current"
# ]
#
# ordinal_imputer = SimpleImputer(strategy="most_frequent")
# categorical_data[ordinal_cols] = ordinal_imputer.fit_transform(
#     categorical_data[ordinal_cols]
# )
#
# ordinal_encoder = OrdinalEncoder(
#     categories=[
#         education_order,
#         income_order,
#         smoking_order
#     ]
# )
#
# categorical_data[ordinal_cols] = ordinal_encoder.fit_transform(
#     categorical_data[ordinal_cols]
# )
#
# print("\nOrdinal encoded values (first 5 rows):")
# print(categorical_data[ordinal_cols].head())
#
# onehot_imputer = SimpleImputer(strategy="most_frequent")
# categorical_data[onehot_cols] = onehot_imputer.fit_transform(
#     categorical_data[onehot_cols]
# )
#
# onehot_encoder = OneHotEncoder(
#     handle_unknown="ignore",
#     sparse_output=False
# )
#
# categorical_data_ohe = onehot_encoder.fit_transform(
#     categorical_data[onehot_cols]
# )
#
# ohe_features = onehot_encoder.get_feature_names_out(onehot_cols)
# categorical_data_ohe = pd.DataFrame(
#     categorical_data_ohe,
#     columns=ohe_features,
#     index=categorical_data.index
# )
#
# print("\nOne-hot encoded feature names:")
# print(list(ohe_features))
#
# target_imputer = SimpleImputer(strategy="most_frequent")
# categorical_data[target_cols] = target_imputer.fit_transform(
#     categorical_data[target_cols]
# )
#
# target_encoder = TargetEncoder(smoothing=10)
# categorical_data_target = target_encoder.fit_transform(
#     categorical_data[target_cols],
#     y
# )
#
# print("\nTarget encoded ethnicity (first 5 rows):")
# print(categorical_data_target.head())
#
# categorical_data.drop(columns=onehot_cols + target_cols, inplace=True)
#
# final_categorical_data = pd.concat(
#     [categorical_data, categorical_data_ohe, categorical_data_target],
#     axis=1
# )
#
# print("\nFinal encoded dataset shape:")
# print(final_categorical_data.shape)
#
# print("\nFinal encoded dataset preview:")
# print(final_categorical_data.head())
#
# non_categorical_columns = data.drop(
#     columns=categorical_columns + ["diagnosed_diabetes"]
# )
#
# updated_data = pd.concat(
#     [non_categorical_columns, final_categorical_data, y],
#     axis=1
# )
#
# print("\nUpdated dataset shape:")
# print(updated_data.shape)
#
# print("\nUpdated dataset preview:")
# print(updated_data.head())
#
#
# updated_data.to_csv("updated_train_encoded.csv", index=False)
# print("\nUpdated encoded dataset saved as updated_train_encoded.csv")
#
# # ---- STEP 1: Prepare X and y ----
# X = updated_data.drop(columns=[y.name])
# y_final = updated_data[y.name]
#
# # ---- STEP 2: FILTER METHOD (Mutual Information) ----
# mi_scores = mutual_info_classif(X, y_final, random_state=42)
# mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
#
# top_mi_features = mi_ranking.head(12).index.tolist()  # reduced pool
# print("\nTop MI features:", top_mi_features)
#
# X_mi = X[top_mi_features]
#
# # ---- STEP 3: SINGLE RandomForest Fit ----
# rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
# rf.fit(X_mi, y_final)
#
# importances = pd.Series(rf.feature_importances_, index=X_mi.columns).sort_values(ascending=False)
# print("\nRF Importances:\n", importances)
#
# # ---- STEP 4: Embedded Selection once ----
# sfm = SelectFromModel(rf, threshold="median", prefit=True)
# sfm_selected = X_mi.columns[sfm.get_support()].tolist()
# print("\nSelected by SFM:", sfm_selected)
#
# # ---- STEP 5: Final Dataset ----
# X_final_selected = X[sfm_selected]
# final_data = pd.concat([X_final_selected, y_final], axis=1)
#
# print("\nFinal shape:", final_data.shape)
# final_data.to_csv("final_train_selected.csv", index=False)

# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from category_encoders.target_encoder import TargetEncoder
import sys

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

data = pd.read_csv(INPUT_PATH)

categorical_columns = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status"
]

categorical_data = data[categorical_columns].copy()
y = data["diagnosed_diabetes"]

ordinal_cols = ["education_level", "income_level", "smoking_status"]
onehot_cols = ["gender", "employment_status"]
target_cols = ["ethnicity"]

education_order = ["No formal", "Highschool", "Graduate", "Postgraduate"]
income_order = ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"]
smoking_order = ["Never", "Former", "Current"]

categorical_data[ordinal_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
    categorical_data[ordinal_cols]
)

categorical_data[ordinal_cols] = OrdinalEncoder(
    categories=[education_order, income_order, smoking_order]
).fit_transform(categorical_data[ordinal_cols])

categorical_data[onehot_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
    categorical_data[onehot_cols]
)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe_data = ohe.fit_transform(categorical_data[onehot_cols])
ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(onehot_cols))

categorical_data[target_cols] = SimpleImputer(strategy="most_frequent").fit_transform(
    categorical_data[target_cols]
)

te = TargetEncoder(smoothing=10)
target_df = te.fit_transform(categorical_data[target_cols], y)

categorical_data.drop(columns=onehot_cols + target_cols, inplace=True)

final_cat = pd.concat([categorical_data, ohe_df, target_df], axis=1)
non_cat = data.drop(columns=categorical_columns + ["diagnosed_diabetes"])

updated = pd.concat([non_cat, final_cat, y], axis=1)

X = updated.drop(columns=["diagnosed_diabetes"])
y_final = updated["diagnosed_diabetes"]

mi = mutual_info_classif(X, y_final, random_state=42)
top_features = pd.Series(mi, index=X.columns).sort_values(ascending=False).head(12).index

rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X[top_features], y_final)

sfm = SelectFromModel(rf, threshold="median", prefit=True)
selected = X[top_features].columns[sfm.get_support()]

final_data = pd.concat([X[selected], y_final], axis=1)
final_data.to_csv(OUTPUT_PATH, index=False)

print("Saved processed data →", OUTPUT_PATH)

