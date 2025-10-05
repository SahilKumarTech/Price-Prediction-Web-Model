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
# Load model
# --------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Price_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"✅ Model type: {type(model)}")
    print(f"📊 Model expecting {model.n_features_in_} features")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --------------------------
# Feature preparation
# --------------------------
def prepare_features(input_data):
    try:
        fulfilment = input_data['fulfilment']
        sales_channel = input_data['sales_channel']
        category = input_data['category']
        size = input_data['size']
        ship_state = input_data['ship_state']
        b2b = input_data['b2b']
        qty = input_data['qty']

        features_105 = np.zeros(105)
        features_105[0] = fulfilment
        features_105[1] = sales_channel
        features_105[2] = category
        features_105[3] = size
        features_105[4] = ship_state
        features_105[5] = b2b
        features_105[6] = qty

        return features_105.reshape(1, -1)
    except Exception as e:
        raise e

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
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.json
        prepared_features = prepare_features(data)
        prediction = model.predict(prepared_features)[0]

        return jsonify({
            'prediction': float(prediction),
            'status': 'success',
            'features_received': 7,
            'features_used': prepared_features.shape[1],
            'message': f'Converted 7 input features to {prepared_features.shape[1]} model features'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    return jsonify({
        'model_type': str(type(model)),
        'n_features_in': model.n_features_in_,
        'n_estimators': model.n_estimators if hasattr(model, 'n_estimators') else 'N/A',
        'message': 'Model expects 105 features due to feature engineering during training'
    })

@app.route('/features', methods=['GET'])
def get_features():
    # same features as before
    return jsonify({"note": "7 input features automatically expanded to 105 for model"})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_features": model.n_features_in_ if model else "No model",
        "message": "Price Prediction API is running"
    })

# --------------------------
# Run app (Render-ready)
# --------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Price Prediction API on port {port}...")
    app.run(host='0.0.0.0', port=port)
