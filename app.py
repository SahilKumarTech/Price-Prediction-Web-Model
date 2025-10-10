from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# --------------------------
# Allow cross-origin requests
# --------------------------
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# --------------------------
# Load ML Model safely
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
# Prepare input features (7 total)
# --------------------------
def prepare_features(input_data):
    try:
        fulfilment = float(input_data.get('fulfilment', 0))
        sales_channel = float(input_data.get('sales_channel', 0))
        category = float(input_data.get('category', 0))
        size = float(input_data.get('size', 0))
        ship_state = float(input_data.get('ship_state', 0))
        b2b = float(input_data.get('b2b', 0))
        qty = float(input_data.get('qty', 1))

        # ✅ Only 7 input features — matches trained model
        features = np.array([
            fulfilment,
            sales_channel,
            category,
            size,
            ship_state,
            b2b,
            qty
        ]).reshape(1, -1)

        print(f"🔧 Prepared features: {features}")
        return features

    except Exception as e:
        print(f"❌ Feature preparation error: {e}")
        raise ValueError(f"Feature preparation failed: {str(e)}")


# --------------------------
# Flask Routes
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

        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        print(f"📥 Received data: {data}")

        # Prepare 7 input features
        prepared_features = prepare_features(data)

        # Validate feature dimensions
        if hasattr(model, 'n_features_in_') and prepared_features.shape[1] != model.n_features_in_:
            return jsonify({
                'error': f'Feature dimension mismatch. Model expects {model.n_features_in_}, got {prepared_features.shape[1]}'
            }), 400

        # Predict
        prediction = model.predict(prepared_features)[0]
        prediction_float = float(prediction)
        print(f"🎯 Prediction made: {prediction_float}")

        return jsonify({
            'prediction': prediction_float,
            'status': 'success'
        })

    except ValueError as ve:
        print(f"❌ Validation error: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


# --------------------------
# Health Check Endpoint
# --------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "api_version": "1.0"
    })


# --------------------------
# Run the app
# --------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
