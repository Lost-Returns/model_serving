import tensorflow as tf

def load_model():
    # 모델 로드 및 초기화 로직
    model = tf.keras.models.load_model('C:/grad_proj/mobilenet_224_classifier_1000.h5')
    return model