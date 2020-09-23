import numpy as np
from tensorflow.keras.models import load_model
from myutils.file_processor import get_files
from myutils.image_processor import preprocess_image


def testing_model(
    model_path,
    test_dir_path,
    target_height=0,
    target_width=0
):
    temp_test_list = [test_dir_path]

    test_dir_list = temp_test_list

    model = load_model(model_path)
    model.summary()

    for tdl in test_dir_list:
        test_images_list = get_files(tdl, '.jpg')

        for dir_count, items in enumerate(test_images_list):

            current_dir = items[0]
            images_name_list = items[1]
            images_paths_list = [current_dir + "/" + image_name for image_name in images_name_list]

            count0 = 0
            count1 = 0
            count2 = 0
            for image_path in images_paths_list:
                img = preprocess_image(
                    image_path,
                    target_height=target_height,
                    target_width=target_width,
                    rescale=1/255.0,
                    batch_mode=True
                )
                predictions = model.predict(img)
                print(predictions)
                label = np.argmax(predictions)
                if label == 0:
                    count0 += 1
                elif label == 1:
                    count1 += 1
                elif label == 2:
                    count2 += 1


if __name__ == '__main__':
    test_model_path = "clf_models/model.h5"
    test_dir_path = r"test_dataset"

    testing_model(
        model_path=test_model_path,
        test_dir_path=test_dir_path,
        target_height=512,
        target_width=512
    )
