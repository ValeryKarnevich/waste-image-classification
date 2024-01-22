import scipy
import tensorflow as tf
import tensorflow.keras as keras


def convert_model():
    model_filename = 'resnet50_best_model.h5'

    model = keras.models.load_model(model_filename)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    tflite_model_filename = 'waste-model.tflite'
    with open(tflite_model_filename, 'wb') as f_out:
        f_out.write(tflite_model)

    print(f'''Keras model {model_filename} 
        converted to {tflite_model_filename}''')


if __name__ == '__main__':
    convert_model()