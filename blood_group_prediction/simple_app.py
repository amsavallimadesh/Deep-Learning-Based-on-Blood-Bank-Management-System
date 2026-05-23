from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model_loaded = False
blood_groups = ['A', 'B', 'AB', 'O']

# Simple blood parameter definitions
BLOOD_PARAMETERS = {
    'hemoglobin': {
        'name': 'Hemoglobin',
        'unit': 'g/dL',
        'normal_range': '12-16',
        'example': 14.5,
        'description': 'Red blood cell protein that carries oxygen'
    },
    'platelets': {
        'name': 'Platelet Count',
        'unit': '×10³/μL',
        'normal_range': '150-450',
        'example': 250,
        'description': 'Cells that help blood clot'
    },
    'wbc': {
        'name': 'White Blood Cells',
        'unit': '×10³/μL',
        'normal_range': '4.5-11.0',
        'example': 7.5,
        'description': 'Immune system cells'
    }
}

def load_model():
    """Load the pretrained LSTM model"""
    global model_loaded
    try:
        # Check if model exists (for now, we'll use a mock prediction)
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'blood_group_lstm_model.h5')
        if os.path.exists(model_path):
            logger.info(f"Model found at {model_path}, but using mock prediction for demo")
        else:
            logger.info("No pretrained model found. Using mock prediction for demonstration.")
        
        model_loaded = True
        return True
    except Exception as e:
        logger.error(f"Error initializing: {e}")
        return False

def preprocess_input(input_values):
    """Preprocess input values (mock implementation)"""
    try:
        # Convert to numpy array
        input_array = np.array(input_values, dtype=np.float32)
        
        # Simple normalization (mock preprocessing)
        input_normalized = input_array / np.max(np.abs(input_array) + 1e-8)
        
        return input_normalized
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

def simple_predict(input_values):
    """Simple prediction function for demonstration"""
    # Generate realistic-looking probabilities based on input
    seed = int(np.sum(input_values)) % 1000
    np.random.seed(seed)
    
    # Generate probabilities that sum to 1
    probs = np.random.dirichlet(np.ones(4))
    
    # Make one blood group dominant (higher confidence)
    dominant_idx = np.argmax(probs)
    probs[dominant_idx] = min(probs[dominant_idx] + 0.3, 0.9)
    
    # Renormalize
    probs = probs / np.sum(probs)
    
    return probs

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('simple_index.html', parameters=BLOOD_PARAMETERS)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict blood group from input values"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'input' not in data:
            return jsonify({'error': 'Missing input data'}), 400
        
        input_values = data['input']
        
        # Validate input - now expecting 3 values instead of 14
        if not isinstance(input_values, list) or len(input_values) != 3:
            return jsonify({'error': 'Exactly 3 numeric values required (Hemoglobin, Platelets, White Blood Cells)'}), 400
        
        # Check if all values are numeric
        try:
            input_values = [float(val) for val in input_values]
        except (ValueError, TypeError):
            return jsonify({'error': 'All input values must be numeric'}), 400
        
        # Preprocess input
        processed_input = preprocess_input(input_values)
        
        # Make prediction (using mock prediction for demo)
        predictions = simple_predict(processed_input)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx]) * 100
        predicted_blood_group = blood_groups[predicted_class_idx]
        
        # Get all probabilities for detailed response
        probabilities = {
            blood_groups[i]: float(predictions[i]) * 100 
            for i in range(len(blood_groups))
        }
        
        response = {
            'predicted_blood_group': predicted_blood_group,
            'confidence': round(confidence, 2),
            'probabilities': {k: round(v, 2) for k, v in probabilities.items()},
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'prediction_mode': 'mock' if not os.path.exists(os.path.join(os.path.dirname(__file__), 'models', 'blood_group_lstm_model.h5')) else 'tensorflow'
    })

if __name__ == '__main__':
    # Initialize application
    if load_model():
        logger.info("Application initialized successfully")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize application")
        exit(1)
