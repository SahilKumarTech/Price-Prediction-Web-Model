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
# Load model safely
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


def prepare_features(input_data):
    try:
        fulfilment = float(input_data.get('fulfilment', 0))
        sales_channel = float(input_data.get('sales_channel', 0))
        category = float(input_data.get('category', 0))
        size = float(input_data.get('size', 0))
        ship_state = float(input_data.get('ship_state', 0))
        b2b = float(input_data.get('b2b', 0))
        qty = float(input_data.get('qty', 1))

        features_105 = np.zeros(105)
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

        # JSON data expected from frontend
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        print(f"📥 Received data: {data}")

        prepared_features = prepare_features(data)

        if prepared_features.shape[1] != model.n_features_in_:
            return jsonify({
                'error': f'Feature dimension mismatch. Model expects {model.n_features_in_}, got {prepared_features.shape[1]}'
            }), 400

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


# Health endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "api_version": "1.0"
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Render uses host 0.0.0.0
    app.run(host='0.0.0.0', port=port, debug=False)
