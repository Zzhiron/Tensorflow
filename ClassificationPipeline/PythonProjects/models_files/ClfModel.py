
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import losses


class ClfModel:
    def __init__(self):
        self.input_data = None
        self.model = None
        self.output = None

    def build(self, target_height, target_width, target_channel, compile_model_flag=False):
        self.input_data = Input(shape=(target_height, target_width, target_channel), name='input_layer')
        self.output = self.call()
        self.model = Model(self.input_data, self.output)
        if compile_model_flag:
            self.compile_model()
        return self.model

    def compile_model(self):
        optimizer_adam = optimizers.Adam()
        loss_function = losses.BinaryCrossentropy()
        self.model.compile(optimizer=optimizer_adam,
                           loss=loss_function,
                           metrics=["acc"])

    def call(self):
        padding = "same"
        x = Conv2D(32, kernel_size=(3, 3), strides=2, activation='relu', padding=padding,
                   name='conv1')(self.input_data)
        x = Conv2D(32, kernel_size=(3, 3), strides=2, activation='relu', padding=padding,
                   name='conv2')(x)
        x = Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu', padding=padding,
                   name='conv3')(x)
        x = Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu', padding=padding,
                   name='conv4')(x)
        x = Conv2D(128, kernel_size=(8, 8), strides=4, activation='relu', padding=padding,
                   name='conv5')(x)
        x = Conv2D(128, kernel_size=(8, 8), strides=4, activation='relu', padding=padding,
                   name='conv6')(x)
        x = Flatten()(x)
        x = Dense(256, activation="relu", name="fc")(x)
        predictions = Dense(2, activation="sigmoid", name="output_layer")(x)
        return predictions


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model_init = ClfModel()
    test_model = model_init.build(512, 640, 3, compile_model_flag=True)
    test_model.summary()