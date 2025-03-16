from flask import Flask, render_template, request, jsonify
import os
import PyPDF2
import pickle
import re
import numpy as np
import json
import psycopg2
from psycopg2.extras import DictCursor

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# PostgreSQL Connection Configuration
DB_CONFIG = {
    'dbname': 'postgres',  # Connect to default postgres database first
    'user': 'postgres',
    'password': 'priyanshu@789',
    'host': 'localhost',
    'port': '5432'
}

DB_NAME = 'medical_fraud'  # The database we want to use

def create_database():
    """Create the database if it doesn't exist"""
    # Connect to default postgres database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True  # Set autocommit to True to create database
        
        with conn.cursor() as cur:
            # Check if our database exists
            cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (DB_NAME,))
            exists = cur.fetchone()
            
            if not exists:
                print(f"Creating database {DB_NAME}")
                cur.execute(f"CREATE DATABASE {DB_NAME}")
                print(f"Database {DB_NAME} created successfully")
            else:
                print(f"Database {DB_NAME} already exists")
    except Exception as e:
        print(f"Error creating database: {str(e)}")
    finally:
        if conn:
            conn.close()

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        # Use the medical_fraud database for normal operations
        config = DB_CONFIG.copy()
        config['dbname'] = DB_NAME
        conn = psycopg2.connect(**config)
        return conn
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return None

def create_predictions_table():
    """Create the predictions table if it doesn't exist"""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    bene_id VARCHAR(50),
                    provider VARCHAR(255),
                    prediction_result BOOLEAN,
                    prediction_probability FLOAT,
                    file_name VARCHAR(255),
                    extracted_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                conn.commit()
            print("Predictions table created or already exists")
        except Exception as e:
            print(f"Error creating table: {str(e)}")
        finally:
            conn.close()

# Create the database and table when the app starts
try:
    create_database()
    create_predictions_table()
except Exception as e:
    print(f"Setup error: {str(e)}")

# Load the XGBoost model
def load_model():
    try:
        model_path = r".\weights\xgboost_model.pkl"
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("XGBoost model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Global variable for the model
xgb_model = load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Check if the file is a PDF
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Extract text from PDF using PyPDF2
        extracted_text = extract_text_from_pdf(file_path)
        
        # Return the extracted text directly for UI display
        return jsonify({"filename": file.filename, "extracted_text": extracted_text})
    
    except Exception as e:
        # Log the error
        print(f"Error processing request: {str(e)}")
        # Return JSON even in case of error
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
def predict_from_text():
    try:
        # Get text input from the form
        input_text = request.form.get('extracted_text', '')
        filename = request.form.get('filename', 'unknown_file')
        
        if not input_text:
            return jsonify({"error": "No text input provided"}), 400
        
        # Parse the input text into a dictionary with field names and values
        parsed_data = parse_invoice_data(input_text)
        
        # Extract BeneID and Provider from parsed data
        bene_id = extract_bene_id(parsed_data, input_text)
        provider = extract_provider(parsed_data, input_text)
        
        # Extract values for model prediction
        feature_values = extract_feature_values(parsed_data)
        
        # Make predictions if model is loaded and we have values
        if xgb_model is not None and feature_values:
            # Prepare features for the model
            features = prepare_features(feature_values)
            
            # Get probability of fraud (class 1)
            prediction_proba = xgb_model.predict_proba(features)[:, 1]

            # Set a custom threshold (e.g., 0.4 instead of 0.5)
            threshold = 0.4
            prediction = (prediction_proba >= threshold).astype(int)
            prediction_proba = xgb_model.predict_proba(features) if hasattr(xgb_model, 'predict_proba') else None
            
            # Format the prediction result
            prediction_result = {
                "prediction_class": int(prediction[0]),
                "is_fraud": bool(prediction[0] == 1),  # Assuming 1 means fraud
                "bene_id": bene_id,
                "provider": provider
            }
            
            # Add probability if available
            fraud_probability = None
            if prediction_proba is not None:
                fraud_probability = float(prediction_proba[0][1])  # Probability of fraud
                prediction_result["prediction_probability"] = fraud_probability
            
            # Save prediction to database
            save_prediction_to_db(bene_id, provider, prediction_result["is_fraud"], 
                                 fraud_probability, filename, input_text)
            
            return jsonify({
                "parsed_data": parsed_data,
                "feature_values": feature_values,
                "prediction": prediction_result
            })
        else:
            if xgb_model is None:
                return jsonify({"error": "Model not loaded properly"}), 500
            else:
                return jsonify({"error": "No valid values found in input"}), 400
    
    except Exception as e:
        # Log the error
        print(f"Error processing request: {str(e)}")
        # Return JSON even in case of error
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def extract_bene_id(parsed_data, text):
    """Extract Beneficiary ID from parsed data or raw text"""
    # Try common field names for Beneficiary ID
    bene_id_fields = ['BeneID', 'Bene ID', 'Beneficiary ID', 'BeneficiaryID', 'Patient ID', 'PatientID']
    
    # Check parsed data first
    for field in bene_id_fields:
        if field in parsed_data:
            return str(parsed_data[field])
    
    # If not found in parsed data, try regex patterns
    patterns = [
        r'(?i)bene(?:\s*)?id[:\s]+([A-Z0-9]+)',
        r'(?i)beneficiary(?:\s*)?id[:\s]+([A-Z0-9]+)',
        r'(?i)patient(?:\s*)?id[:\s]+([A-Z0-9]+)',
        r'(?i)ID[:\s]+([A-Z0-9]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return "Unknown"

def extract_provider(parsed_data, text):
    """Extract Provider information from parsed data or raw text"""
    # Try common field names for Provider
    provider_fields = ['Provider', 'Provider Name', 'ProviderName', 'Doctor', 'Physician', 'Hospital']
    
    # Check parsed data first
    for field in provider_fields:
        if field in parsed_data:
            return str(parsed_data[field])
    
    # If not found in parsed data, try regex patterns
    patterns = [
        r'(?i)provider(?:\s*)?(?:name)?[:\s]+([A-Za-z0-9\s&.,]+)',
        r'(?i)doctor[:\s]+([A-Za-z0-9\s&.,]+)',
        r'(?i)hospital[:\s]+([A-Za-z0-9\s&.,]+)',
        r'(?i)physician[:\s]+([A-Za-z0-9\s&.,]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return "Unknown"

def save_prediction_to_db(bene_id, provider, is_fraud, probability, file_name, extracted_text):
    """Save prediction results to PostgreSQL database"""
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database when saving prediction")
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                '''
                INSERT INTO predictions 
                (bene_id, provider, prediction_result, prediction_probability, file_name, extracted_text)
                VALUES (%s, %s, %s, %s, %s, %s)
                ''',
                (bene_id, provider, is_fraud, probability, file_name, extracted_text)
            )
            conn.commit()
            print(f"Prediction saved to database for BeneID: {bene_id}, Provider: {provider}")
            return True
    except Exception as e:
        print(f"Database error when saving prediction: {str(e)}")
        return False
    finally:
        conn.close()

def parse_invoice_data(text):
    """
    Parse the invoice data that is formatted with field names and values separated by colons.
    Returns a dictionary with field names as keys and values.
    """
    parsed_data = {}
    
    # Handle the case where text is just a string of values
    if ':' not in text:
        # Try to split by spaces and create a simple dictionary
        values = text.strip().split()
        for i, val in enumerate(values):
            try:
                parsed_data[f"feature_{i}"] = float(val)
            except ValueError:
                parsed_data[f"feature_{i}"] = val
        return parsed_data
    
    # Split the text into lines
    lines = text.split('\n')
    
    # Process each line
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Check if the line contains a colon
        if ':' in line:
            # Split the line by colon
            parts = line.split(':', 1)  # Split on first colon only
            
            if len(parts) == 2:
                field_name = parts[0].strip()
                field_value = parts[1].strip()
                
                # Try to convert to float if possible
                try:
                    field_value = float(field_value)
                except ValueError:
                    # Keep as string if not a number
                    pass
                    
                parsed_data[field_name] = field_value
        else:
            # Split by spaces and treat as separate values
            values = line.strip().split()
            for i, val in enumerate(values):
                key = f"value_{i}"
                try:
                    parsed_data[key] = float(val)
                except ValueError:
                    parsed_data[key] = val
    
    return parsed_data

def extract_feature_values(parsed_data):
    """
    Extract values from the parsed data to use as features for the model.
    Returns a list of numeric values.
    """
    # Filter out only numeric values
    feature_values = []
    
    for key, value in parsed_data.items():
        if isinstance(value, (int, float)):
            feature_values.append(value)
    
    return feature_values

def prepare_features(feature_values):
    """
    Convert feature values to the format required by the model.
    """
    # Expected number of features for the model
    # Assuming the model was trained with 53 features
    expected_feature_count = 53
    
    # Ensure we have the right number of features
    if len(feature_values) < expected_feature_count:
        feature_values = feature_values + [0.0] * (expected_feature_count - len(feature_values))
    elif len(feature_values) > expected_feature_count:
        feature_values = feature_values[:expected_feature_count]
    
    # Return as numpy array in the shape expected by XGBoost
    return np.array([feature_values])

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        # Open the PDF file in binary mode
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get the number of pages
            num_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text:  # Check if text was extracted
                    text += page_text + "\n"
                else:
                    text += f"[No text could be extracted from page {page_num + 1}]\n"
        
        return text.strip()
    except Exception as e:
        raise Exception(f"PDF processing error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)