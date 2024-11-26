from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
import ast

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

class UserBasedFraudDetection:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.isolation_forest = None
        self.autoencoder = None
        self.user_patterns = {}
        
        # Columns to exclude from model training
        self.exclude_columns = ['sessionId', 'orgId', 'appId', 'publicIPAddress', 
                              'localIPAddress', 'timeDifferenceInMilliseconds']
        
        # Numerical columns
        self.numerical_columns = [
            'dbVersion', 'longitude', 'latitude', 'batteryLevel',
            'xCoordinate_min', 'xCoordinate_max', 'xCoordinate_mean',
            'yCoordinate_min', 'yCoordinate_max', 'yCoordinate_mean'
        ]
        
        # Categorical columns to encode
        self.categorical_columns = [
            'networkType', 'timezone', 'language', 'displayResolution',
            'powerSource', 'device_deviceType_name', 'device_devicePlatform_name',
            'device_os'
        ]
        
    def preprocess_time_features(self, df):
        """Extract time-based features"""
        df = df.copy()
        
        # Convert to datetime if not already
        df['startTime'] = pd.to_datetime(df['startTime'])
        df['endTime'] = pd.to_datetime(df['endTime'])
        
        # Extract time features
        df['hour_of_day'] = df['startTime'].dt.hour
        df['day_of_week'] = df['startTime'].dt.dayofweek
        df['session_duration'] = (df['endTime'] - df['startTime']).dt.total_seconds()
        
        # Drop original time columns
        df = df.drop(['startTime', 'endTime'], axis=1)
        
        return df
    
    def encode_categorical_features(self, df, training=True):
        """Encode categorical features using LabelEncoder"""
        df = df.copy()
        
        for column in self.categorical_columns:
            if training:
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column].astype(str))
            else:
                # Initialize label encoder if not exists
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    self.label_encoders[column].fit(df[column].astype(str))
                
                # Handle unseen categories
                le = self.label_encoders[column]
                df[column] = df[column].astype(str)
                df[column] = df[column].map(lambda x: -1 if x not in le.classes_ else le.transform([x])[0])
        
        return df

    def preprocess_data(self, df, training=True):
        """Preprocess the data for model input"""
        try:
            logging.info(f"Columns before preprocessing: {df.columns.tolist()}")
            
            # Convert numerical columns to float
            for col in self.numerical_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Process time features
            df = self.preprocess_time_features(df)
            logging.info(f"Columns after time feature processing: {df.columns.tolist()}")
            
            # Encode categorical features
            df = self.encode_categorical_features(df, training)
            logging.info(f"Columns after categorical encoding: {df.columns.tolist()}")
            
            # Select features for model
            features_for_model = (self.numerical_columns + self.categorical_columns + 
                                ['hour_of_day', 'day_of_week', 'session_duration'])
            
            # Initialize missing columns with default values
            for col in features_for_model:
                if col not in df.columns:
                    if col in self.numerical_columns:
                        df[col] = 0.0
                    else:
                        df[col] = -1
            
            # Scale numerical features
            if training:
                self.scaler.fit(df[features_for_model])
            scaled_features = self.scaler.transform(df[features_for_model])
            
            return pd.DataFrame(scaled_features, columns=features_for_model)
            
        except Exception as e:
            logging.error(f"Error in preprocess_data: {e}")
            raise

    def split_user_data(self, df, test_size=5):
        """Split data into train and test keeping test_size points per user"""
        train_data = []
        test_data = []
        
        for email in df['appUser_email'].unique():
            user_data = df[df['appUser_email'] == email]
            
            # Randomly select test points
            test_indices = np.random.choice(user_data.index, size=min(test_size, len(user_data)), 
                                          replace=False)
            train_indices = user_data.index.difference(test_indices)
            
            train_data.append(df.loc[train_indices])
            test_data.append(df.loc[test_indices])
        
        return pd.concat(train_data), pd.concat(test_data)

    def train_models(self, df):
        """Train models on the training data"""
        # Split data
        train_df, self.test_df = self.split_user_data(df)
        
        # Preprocess training data
        train_processed = self.preprocess_data(train_df, training=True)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(train_processed)
        
        # Store score ranges for normalization
        train_if_scores = -self.isolation_forest.score_samples(train_processed)
        self.if_min = train_if_scores.min()
        self.if_max = train_if_scores.max()
        
        # Build and train autoencoder
        input_dim = train_processed.shape[1]
        self.autoencoder = self.build_autoencoder(input_dim)
        
        self.autoencoder.fit(
            train_processed,
            train_processed,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Store reconstruction error ranges
        predictions = self.autoencoder.predict(train_processed)
        reconstruction_errors = np.mean(np.square(train_processed - predictions), axis=1)
        self.ae_min = reconstruction_errors.min()
        self.ae_max = reconstruction_errors.max()
        
        # Learn user patterns
        self.learn_user_patterns(train_df)

    def learn_user_patterns(self, df):
        """Learn normal patterns for each user"""
        for email in df['appUser_email'].unique():
            user_data = df[df['appUser_email'] == email]
            
            self.user_patterns[email] = {
                'numerical_stats': {},
                'categorical_patterns': {},
                'time_patterns': {}
            }
            
            # Numerical patterns
            for col in self.numerical_columns:
                self.user_patterns[email]['numerical_stats'][col] = {
                    'mean': user_data[col].mean(),
                    'std': user_data[col].std(),
                    'q1': user_data[col].quantile(0.25),
                    'q3': user_data[col].quantile(0.75)
                }
            
            # Categorical patterns
            for col in self.categorical_columns:
                self.user_patterns[email]['categorical_patterns'][col] = (
                    user_data[col].value_counts(normalize=True)
                )
            
            # Time patterns
            time_data = self.preprocess_time_features(user_data)
            for col in ['hour_of_day', 'day_of_week', 'session_duration']:
                self.user_patterns[email]['time_patterns'][col] = {
                    'mean': time_data[col].mean(),
                    'std': time_data[col].std()
                }
    logging.basicConfig(level=logging.INFO)

    def detect_anomalies(self, row):
        """
        Detect specific anomalies in user behavior by comparing against learned patterns
        Returns tuple of (list of anomalies, list of justifications)
        """
        email = str(row['appUser_email'])
        logging.info(f"Checking patterns for user: {email}")

        anomalies = []
        justifications = []
        
        # Skip if user patterns not found (new user)
        if email not in self.user_patterns:
            return (
                ["Unknown User Pattern"], 
                ["No historical data available for this user"]
            )
        
        user_pattern = self.user_patterns[email]
        
        # Check numerical features
        for col in self.numerical_columns:
            try:
                value = float(row[col])
                stats = user_pattern['numerical_stats'][col]
                
                # Check if value is outside 3 standard deviations
                if stats['std'] != 0:
                    z_score = abs((value - stats['mean']) / stats['std'])
                    if z_score > 3:
                        anomalies.append(f"Unusual {col}")
                        justifications.append(
                            f"{col} value ({value:.2f}) is {z_score:.2f} standard deviations "
                            f"from user's mean ({stats['mean']:.2f})"
                        )
                
                # Check if value is outside IQR
                iqr = stats['q3'] - stats['q1']
                if iqr > 0:
                    if value < (stats['q1'] - 1.5 * iqr) or value > (stats['q3'] + 1.5 * iqr):
                        anomalies.append(f"Outlier {col}")
                        justifications.append(
                            f"{col} value ({value:.2f}) is outside the typical range "
                            f"[{stats['q1']:.2f}, {stats['q3']:.2f}]"
                        )
            except Exception as e:
                print(f"Error processing numerical feature {col}: {str(e)}")
                continue
        
        # Check categorical features
        for col in self.categorical_columns:
            try:
                value = str(row[col])
                patterns = user_pattern['categorical_patterns'][col]
                
                # If value never seen before or rarely used by user
                if value not in patterns or patterns[value] < 0.1:
                    anomalies.append(f"Unusual {col}")
                    justifications.append(
                        f"User rarely or never uses {col}={value} "
                        f"(historical frequency: {patterns.get(value, 0):.2%})"
                    )
            except Exception as e:
                print(f"Error processing categorical feature {col}: {str(e)}")
                continue
        
        # Check time patterns
        time_data = self.preprocess_time_features(pd.DataFrame([row]))
        for col in ['hour_of_day', 'day_of_week', 'session_duration']:
            try:
                value = float(time_data[col].iloc[0])
                stats = user_pattern['time_patterns'][col]
                
                # Check if time pattern is unusual (outside 2 standard deviations)
                if stats['std'] != 0:
                    z_score = abs((value - stats['mean']) / stats['std'])
                    if z_score > 2:
                        anomalies.append(f"Unusual {col}")
                        justifications.append(
                            f"{col} ({value}) is unusual for this user "
                            f"(typically {stats['mean']:.1f} ± {2*stats['std']:.1f})"
                        )
            except Exception as e:
                print(f"Error processing time feature {col}: {str(e)}")
                continue
        
        return anomalies, justifications

    def evaluate_test_data(self):
        """Evaluate all test data points"""
        results = []
        for _, row in self.test_df.iterrows():
            result = self.analyze_session(row)
            results.append(result)
        return pd.DataFrame(results)

    def analyze_session(self, row):
        try:
            """Analyze a single session for threats"""
            logging.info(f"Analyzing row: {row}")
            email = row['appUser_email']
            
            # Prepare data for model scoring
            row_df = pd.DataFrame([row])
            logging.info(f"Preprocessed row DataFrame: {row_df}")

            row_processed = self.preprocess_data(row_df, training=False).values
            logging.info(f"Processed row for model: {row_processed}")  # Ensure NumPy array
            
            # Get model scores
            if_score = float(-self.isolation_forest.score_samples(row_processed)[0])
            logging.info(f"Isolation Forest score: {if_score}")

            ae_pred = self.autoencoder.predict(row_processed)
            logging.info(f"Autoencoder prediction: {ae_pred}")
            
            # Ensure shapes are compatible
            assert row_processed.shape == ae_pred.shape, f"Shape mismatch: {row_processed.shape} vs {ae_pred.shape}"
            
            # Calculate Autoencoder reconstruction error
            ae_score = float(np.mean(np.square(row_processed - ae_pred)))  # Ensure floats
            
            # Normalize scores
            if_score_norm = float((if_score - self.if_min) / (self.if_max - self.if_min))
            ae_score_norm = float((ae_score - self.ae_min) / (self.ae_max - self.ae_min))
            
            # Combined threat score
            threat_score = 0.5 * if_score_norm + 0.5 * ae_score_norm
            
            # Get anomalies and justifications
            anomalies, justifications = self.detect_anomalies(row)
            
            # Determine threat level
            threat_level = self.get_threat_level(threat_score, len(anomalies))
            
            return {
                "session_id": str(row['sessionId']),
                "user_email": str(email),
                "threat_signals": anomalies,
                "justifications": justifications,
                "threat_score": threat_score,
                "threat_level": threat_level
            }
        except KeyError as e:
            logging.error(f"Missing column: {e}")
            raise
        
        except Exception as e:
            logging.error(f"Error in analyze_session: {e}")
            raise

            
    def get_threat_level(self, threat_score, num_anomalies):
        """Determine threat level based on score and number of anomalies"""
        # Ensure we're working with scalar values
        if isinstance(threat_score, (pd.Series, np.ndarray)):
            threat_score = float(threat_score.iloc[0] if isinstance(threat_score, pd.Series) else threat_score[0])
        if isinstance(num_anomalies, (pd.Series, np.ndarray)):
            num_anomalies = int(num_anomalies.iloc[0] if isinstance(num_anomalies, pd.Series) else num_anomalies[0])
        
        # Convert to native Python types if needed
        threat_score = float(threat_score)
        num_anomalies = int(num_anomalies)
        
        if threat_score > 0.8 and num_anomalies >= 3:
            return "Critical"
        elif threat_score > 0.6 or num_anomalies >= 2:
            return "High"
        elif threat_score > 0.4 or num_anomalies >= 1:
            return "Medium"
        else:
            return "Low"

    def build_autoencoder(self, input_dim):
        """Build and compile autoencoder model"""
        input_layer = Input(shape=(input_dim,))
        
        # Encoder
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Build model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    


# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent user sessions",
    version="1.0.0"
)
# Create and train the model at startup
print("Initializing and training the fraud detection system...")
df = pd.read_csv('synthetic_behavioral_data.csv')
test_data = df.groupby('appUser_email').apply(lambda x: x.sample(n=1)).reset_index(drop=True)
train_data = df.drop(test_data.index)

# Process coordinates
train_data['xCoordinate'] = train_data['xCoordinate'].apply(ast.literal_eval)
train_data['yCoordinate'] = train_data['yCoordinate'].apply(ast.literal_eval)

train_data['xCoordinate_min'] = train_data['xCoordinate'].apply(min)
train_data['xCoordinate_max'] = train_data['xCoordinate'].apply(max)
train_data['xCoordinate_mean'] = train_data['xCoordinate'].apply(lambda x: sum(x) / len(x))

train_data['yCoordinate_min'] = train_data['yCoordinate'].apply(min)
train_data['yCoordinate_max'] = train_data['yCoordinate'].apply(max)
train_data['yCoordinate_mean'] = train_data['yCoordinate'].apply(lambda x: sum(x) / len(x))

# Initialize and train the model globally
model = UserBasedFraudDetection()
model.train_models(train_data)
print("Model training completed!")

class SessionData(BaseModel):
    sessionId: str
    startTime: str
    endTime: Optional[str]
    device: Dict[str, Any]
    appUser: Dict[str, Any]
    publicIPAddress: str
    networkType: str
    longitude: Optional[float]
    latitude: Optional[float]
    timezone: str
    language: str
    displayResolution: str
    batteryLevel: str
    powerSource: str
    xCoordinate: List[int]
    yCoordinate: List[int]

class PredictionResponse(BaseModel):
    session_id: str
    user_email: str
    threat_signals: List[str]
    justifications: List[str]
    threat_score: float
    threat_level: str

def preprocess_session_data(session_data: SessionData) -> pd.DataFrame:
    """Convert API input data to DataFrame format matching training data"""
    # Convert SessionData to dict
    data_dict = session_data.model_dump()
    
    # Flatten nested dictionaries
    flattened_data = {
        "sessionId": data_dict["sessionId"],
        "startTime": data_dict["startTime"],
        "endTime": data_dict["endTime"],
        "publicIPAddress": data_dict["publicIPAddress"],
        "networkType": data_dict["networkType"],
        "longitude": data_dict["longitude"],
        "latitude": data_dict["latitude"],
        "timezone": data_dict["timezone"],
        "language": data_dict["language"],
        "displayResolution": data_dict["displayResolution"],
        "batteryLevel": data_dict["batteryLevel"],
        "powerSource": data_dict["powerSource"],
        
        # Device related fields
        "device_deviceType_name": data_dict["device"]["deviceType_name"],
        "device_devicePlatform_name": data_dict["device"]["devicePlatform_name"],
        "device_os": data_dict["device"]["os"],
        
        # AppUser related fields
        "appUser_email": data_dict["appUser"]["email"],
    }
    
    # Add coordinate statistics
    x_coords = data_dict["xCoordinate"]
    y_coords = data_dict["yCoordinate"]
    
    flattened_data.update({
        "xCoordinate_min": min(x_coords),
        "xCoordinate_max": max(x_coords),
        "xCoordinate_mean": sum(x_coords) / len(x_coords),
        "yCoordinate_min": min(y_coords),
        "yCoordinate_max": max(y_coords),
        "yCoordinate_mean": sum(y_coords) / len(y_coords)
    })
    
    # Create DataFrame
    df = pd.DataFrame([flattened_data])
    
    return df


class Device(BaseModel):
    deviceType_name: str
    devicePlatform_name: str
    os: str

class AppUser(BaseModel):
    email: str

class SessionData(BaseModel):
    sessionId: str
    startTime: str
    endTime: str
    device: Device
    appUser: AppUser
    publicIPAddress: str
    networkType: str
    longitude: float
    latitude: float
    timezone: str
    language: str
    displayResolution: str
    batteryLevel: str
    powerSource: str
    xCoordinate: List[float]
    yCoordinate: List[float]

app = FastAPI()

@app.post("/analyze/")
async def analyze_session(data: SessionData):
    try:
        # Log the raw input data for debugging
        logging.info(f"Received data: {data}")
        
        # Preprocess input data
        df = preprocess_session_data(data)
        logging.info(f"Preprocessed DataFrame: {df}")
        
        # Ensure model is initialized
        if model.scaler is None or not hasattr(model.scaler, 'mean_'):
            raise Exception("Model not properly initialized. Please ensure model is trained.")
        
        # Analyze the session using your model
        result = model.analyze_session(df.iloc[0])
        logging.info(f"Analysis result: {result}")
        
        return result
    except Exception as e:
        # Log the error
        logging.error(f"Error processing request: {e}")
        
        # Raise an HTTP exception with a detailed error message
        raise HTTPException(status_code=500, detail=str(e))


# Add a health check endpoint
@app.get("/health")
async def health_check():
    """Check if the model is properly initialized"""
    try:
        is_initialized = (
            model.scaler is not None 
            and hasattr(model.scaler, 'mean_')
            and model.isolation_forest is not None
            and model.autoencoder is not None
        )
        return {
            "status": "healthy" if is_initialized else "not initialized",
            "model_initialized": is_initialized,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)