from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the Hugging Face model
model = pipeline('text-classification', model='./model')

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint required by SageMaker"""
    return jsonify(status="healthy"), 200

@app.route('/invocations', methods=['POST'])
def invocations():
    """Inference endpoint required by SageMaker"""
    data = request.json
    text = data.get('text', '')  # Extract 'text' from JSON payload
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    result = model(text)
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
