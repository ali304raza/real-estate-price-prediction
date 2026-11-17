import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_data():
    df = pd.read_csv("train.csv")

    print("Dataset shape:", df.shape)
    print("\nMissing values:", df.isnull().sum)
    print("\nTarget distribution:\n", df["priceRange"].value_counts())
    print("\nDuplicate rows:", df.duplicated().sum)

    return df

def prepare_data(df):
    X = df.drop(columns=["uid", "description", "priceRange"])
    y = df["priceRange"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("\nTraining shape:", X_train.shape)
    print("\nTesting shape:", X_test.shape)

    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(X):

    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer([("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)])


if __name__ == "__main__":
    df = load_data()
    prepare_data(df)
    
