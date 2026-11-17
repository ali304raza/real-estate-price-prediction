
import joblib
import pandas as pd

model = joblib.load("artifacts/model.joblib")



def predict_price_range(data):

    if isinstance(data, dict):
        data = pd.DataFrame([data])

    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a dictionary or DataFrame")

    required = list(model.named_steps["preprocessor"].feature_names_in_)


    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return model.predict(data[required]).tolist()

if __name__ == "__main__":

    new_property= {
        "city": "austin",
        "homeType": "Single Family",
        "latitude": "30.2672",
        "longitude": -97.7431,
        "garageSpaces": 2,
        "hasSpa": False,
        "yearBuilt": 2015,
        "numOfPatioAndPorchFeatures": 1,
        "lotSizeSqFt": 6500,
        "avgSchoolRating": 7,
        "MedianStudentsPerTeacher": 15,
        "numOfBathrooms": 2,
        "numOfBedrooms": 3
    }

    prediction = predict_price_range(new_property)
    print("New property prediction", prediction[0])


    df = pd.read_csv("train.csv")

    X = df.drop(columns=["uid", "description", "priceRange"])

    predictions = predict_price_range(X)



    print("Total predictions:", len(predictions))
    print("First five predictions:", predictions[:5])

    print("\nInference test completed successfully!")


