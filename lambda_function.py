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


def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)

    float_pred = pred[0].tolist()

    return dict(zip(class_names, float_pred))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result