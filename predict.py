from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


tflite_model_filename = 'waste-model.tflite'

interpreter = tflite.Interpreter(model_path=tflite_model_filename)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = create_preprocessor('resnet50', target_size=(299, 299))

class_names = [
    'battery',
    'glass',
    'metal',
    'organic',
    'paper',
    'plastic'
]

app = Flask('waste')

@app.route('/classify_waste', methods=['POST'])
def predict():
    body = request.get_json()
    url = body['url']

    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)

    float_pred = pred[0].tolist()

    result = dict(zip(class_names, float_pred))
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)