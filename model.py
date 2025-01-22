import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib


def train_model(data_path):
    
    data = pd.read_csv(data_path)
    
    
    X = data[['Temperature', 'Run_Time']]  
    y = data['Downtime_Flag']  
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    
    joblib.dump(model, 'model.pkl')
    
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return {"accuracy": accuracy, "f1_score": f1}


def predict_downtime(input_data):
    
    model = joblib.load('model.pkl')
    
    
    input_df = pd.DataFrame([input_data])
    
    
    prediction = model.predict(input_df)
    confidence = model.predict_proba(input_df).max()
    
    return {"Downtime": "Yes" if prediction[0] == 1 else "No", "Confidence": confidence}
