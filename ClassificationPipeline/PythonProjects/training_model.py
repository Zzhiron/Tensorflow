import os
from models_files.ClfModel import ClfModel
from utils.data_loader import load_data_list
from utils.data_loader import load_data_as_sequence


def training_model(
    weights_path,
    pretrained_weights_path="None",
    batch_size=0,
    epochs_num=0,
    target_height=0,
    target_width=0,
    target_channel=0
):
    model_init = ClfModel()
    model = model_init.build(target_height, target_width, target_channel, compile_model_flag=True)

    train_data_dir = "train_dataset"
    if not (pretrained_weights_path == 'None'):
        if os.path.exists(pretrained_weights_path):
            model.load_weights(pretrained_weights_path)
    model.summary()

    train_paths_list, train_labels_list = load_data_list(train_data_dir)
    train_data = load_data_as_sequence(
        paths_list=train_paths_list,
        labels_list=train_labels_list,
        batch_size=batch_size,
        target_height=target_height,
        target_width=target_width,
        rescale=1 / 255.
    )
    model.fit(
        train_data,
        epochs=epochs_num,
        verbose=2,
    )
    model.save_weights(weights_path)


if __name__ == '__main__':
    models_files_dir = "clf_models/"
    weights_path = models_files_dir + "weights.h5"
    model_path = models_files_dir + "model.h5"

    training_model(
        weights_path=weights_path,
        pretrained_weights_path="None",
        batch_size=16,
        epochs_num=10,
        target_height=512,
        target_width=512,
        target_channel=3
    )