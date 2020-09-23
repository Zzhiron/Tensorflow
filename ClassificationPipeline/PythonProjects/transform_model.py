import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.models import load_model
from models_files.ClfModel import ClfModel


def transform_model(
    models_files_dir,
    target_height=0,
    target_width=0,
    target_channel=0
):
    model_init = ClfModel()
    model = model_init.build(target_height, target_width, target_channel, compile_model_flag=True)

    weights_path = models_files_dir + "weights.h5"
    model_path = models_files_dir + "model.h5"

    model.load_weights(weights_path)
    model.save(model_path, include_optimizer=False)

    model = load_model(model_path)
    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    name = "model.pb"
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        name=name,
        logdir=models_files_dir,
        as_text=False
    )


if __name__ == '__main__':
    models_files_dir = "clf_models/"

    transform_model(
        models_files_dir,
        target_height=512,
        target_width=512,
        target_channel=3
    )