from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="templates")  # Templates folder specify

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
# Load compressed model from local folder
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

        print(f"🔢 Raw features: {[fulfilment, sales_channel, category, size, ship_state, b2b, qty]}")

        # Temporary placeholder for 105 features
        features_105 = np.zeros(105)
        features_105[0] = fulfilment
        features_105[1] = sales_channel
        features_105[2] = category
        features_105[3] = size
        features_105[4] = ship_state
        features_105[5] = b2b
        features_105[6] = qty

        print(f"📈 Prepared {len(features_105)} features for model")
        return features_105.reshape(1, -1)
    except Exception as e:
        print(f"❌ Feature preparation error: {e}")
        raise e

# --------------------------
# Routes
# --------------------------
@app.route('/')
def home():
    # 🔹 Serve HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.json
        print(f"📥 Received data: {data}")

        prepared_features = prepare_features(data)
        print(f"🎯 Making prediction with {prepared_features.shape[1]} features...")

        prediction = model.predict(prepared_features)[0]
        print(f"💰 Prediction: {prediction}")

        return jsonify({
            'prediction': float(prediction),
            'status': 'success',
            'features_received': 7,
            'features_used': prepared_features.shape[1],
            'message': f'Converted 7 input features to {prepared_features.shape[1]} model features'
        })
    except Exception as e:
        print(f"❌ Prediction error: {e}")
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
    features_info = {
        "features": {
            "fulfilment": {"name": "Fulfilment", "options": [{"value": 0, "label": "Amazon.in"}, {"value": 1, "label": "Merchant"}]},
            "sales_channel": {"name": "Sales Channel", "options": [{"value": 0, "label": "Amazon.in"}, {"value": 1, "label": "Non-Amazon"}]},
            "category": {"name": "Category", "options": [{"value": 1, "label": "T-shirt"}, {"value": 2, "label": "Shirt"}, {"value": 3, "label": "Blazzer"}, {"value": 4, "label": "Trousers"}, {"value": 5, "label": "Perfume"}, {"value": 6, "label": "Wallet"}, {"value": 7, "label": "Socks"}, {"value": 8, "label": "Shoes"}, {"value": 9, "label": "Watch"}]},
            "size": {"name": "Size", "options": [{"value": 1, "label": "M"}, {"value": 2, "label": "L"}, {"value": 3, "label": "XL"}, {"value": 4, "label": "XXL"}, {"value": 5, "label": "S"}, {"value": 6, "label": "3XL"}, {"value": 7, "label": "XS"}, {"value": 8, "label": "Free"}, {"value": 9, "label": "6XL"}, {"value": 10, "label": "5XL"}, {"value": 11, "label": "4XL"}]},
            "qty": {"name": "Quantity", "options": [{"value": 1, "label": "1"}, {"value": 0, "label": "0"}, {"value": 2, "label": "2"}, {"value": 3, "label": "3"}, {"value": 4, "label": "4"}, {"value": 5, "label": "5"}, {"value": 9, "label": "9"}, {"value": 15, "label": "15"}]},
            "b2b": {"name": "B2B", "options": [{"value": 0, "label": "False"}, {"value": 1, "label": "True"}]},
            "ship_state": {"name": "Ship State", "options": [{"value": 1, "label": "MAHARASHTRA"}, {"value": 2, "label": "KARNATAKA"}, {"value": 3, "label": "TAMIL NADU"}, {"value": 4, "label": "TELANGANA"}, {"value": 5, "label": "UTTAR PRADESH"}, {"value": 6, "label": "DELHI"}, {"value": 7, "label": "WEST BENGAL"}, {"value": 8, "label": "GUJARAT"}, {"value": 9, "label": "RAJASTHAN"}, {"value": 10, "label": "BIHAR"}, {"value": 11, "label": "NAGALAND"}, {"value": 12, "label": "MIZORAM"}]}
        },
        "note": "Model expects 105 features due to feature engineering. 7 input features are automatically expanded."
    }
    return jsonify(features_info)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_features": model.n_features_in_ if model else "No model",
        "message": "Price Prediction API is running"
    })

if __name__ == '__main__':
    print("🚀 Starting Price Prediction API...")
    if model:
        print(f"📊 Model expects {model.n_features_in_} features")
    app.run(debug=True, host='0.0.0.0', port=5000)

