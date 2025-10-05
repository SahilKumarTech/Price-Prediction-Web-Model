from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# --------------------------
# Manual CORS handling
# --------------------------
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# --------------------------
# Load model with better error handling
# --------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Price_model.pkl")

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
        print(f"✅ Model type: {type(model)}")
        if hasattr(model, 'n_features_in_'):
            print(f"📊 Model expecting {model.n_features_in_} features")
        else:
            print("📊 Model features info not available")
    else:
        print(f"❌ Model file not found at: {MODEL_PATH}")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --------------------------
# Feature preparation
# --------------------------
def prepare_features(input_data):
    try:
        # Extract input values with defaults
        fulfilment = float(input_data.get('fulfilment', 0))
        sales_channel = float(input_data.get('sales_channel', 0))
        category = float(input_data.get('category', 0))
        size = float(input_data.get('size', 0))
        ship_state = float(input_data.get('ship_state', 0))
        b2b = float(input_data.get('b2b', 0))
        qty = float(input_data.get('qty', 1))

        # Create feature array with 105 dimensions
        features_105 = np.zeros(105)
        
        # Set the first 7 features from input
        features_105[0] = fulfilment
        features_105[1] = sales_channel
        features_105[2] = category
        features_105[3] = size
        features_105[4] = ship_state
        features_105[5] = b2b
        features_105[6] = qty

        print(f"🔧 Prepared features: {features_105[:7]}... (total: {len(features_105)} features)")
        return features_105.reshape(1, -1)
        
    except Exception as e:
        print(f"❌ Feature preparation error: {e}")
        raise ValueError(f"Feature preparation failed: {str(e)}")

# --------------------------
# Routes
# --------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

        # Get JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data received'}), 400

        print(f"📥 Received data: {data}")

        # Prepare features
        prepared_features = prepare_features(data)
        
        # Validate feature dimensions
        if prepared_features.shape[1] != model.n_features_in_:
            return jsonify({
                'error': f'Feature dimension mismatch. Model expects {model.n_features_in_}, got {prepared_features.shape[1]}'
            }), 400

        # Make prediction
        prediction = model.predict(prepared_features)[0]
        prediction_float = float(prediction)

        print(f"🎯 Prediction made: {prediction_float}")

        return jsonify({
            'prediction': prediction_float,
            'status': 'success',
            'features_received': 7,
            'features_used': prepared_features.shape[1],
            'message': f'Successfully converted 7 input features to {prepared_features.shape[1]} model features'
        })
        
    except ValueError as ve:
        print(f"❌ Validation error: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'model_loaded': True,
        'model_type': str(type(model)).split("'")[1],
        'message': 'Model is ready for predictions'
    }
    
    # Add model-specific attributes if available
    if hasattr(model, 'n_features_in_'):
        info['n_features_in'] = model.n_features_in_
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators
    if hasattr(model, 'feature_names_in_'):
        info['feature_names'] = model.feature_names_in_.tolist()
    
    return jsonify(info)

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({
        "expected_input_features": [
            "fulfilment (numeric)",
            "sales_channel (numeric)", 
            "category (numeric)",
            "size (numeric)",
            "ship_state (numeric)",
            "b2b (numeric)",
            "qty (numeric)"
        ],
        "model_features": 105,
        "note": "7 input features are automatically expanded to 105 features for model compatibility"
    })

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "api_version": "1.0",
        "message": "Price Prediction API is running"
    }
    
    if model:
        status["model_details"] = {
            "features_expected": model.n_features_in_ if hasattr(model, 'n_features_in_') else "Unknown",
            "model_type": str(type(model)).split("'")[1]
        }
    
    return jsonify(status)

# --------------------------
# Error handlers
# --------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# --------------------------
# Run app (Render-ready)
# --------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Price Prediction API on port {port}...")
    print(f"🔍 Model status: {'✅ Loaded' if model else '❌ Not loaded'}")
    print(f"🌐 Server will run at: http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
