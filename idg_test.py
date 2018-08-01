import os
from utils import *
from keras import backend as k


img_width, img_height = 299, 299
batch_size = 32


def get_data(train_data_dir, validation_data_dir):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # class_mode <- multi_categorical
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='multi_categorical')

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='multi_categorical')
    train_data = train_generator.next()
    val_data = validation_generator.next()

    print("train_data image")
    print(train_data[0])

    print("train_data label")
    print(train_data[1])

    print("-------------------")
    print("val_data image")
    print(val_data[0])

    print("val_data label")
    print(val_data[1])


if __name__ == '__main__':
    data_dir = 'path/to/dataset'
    train_dir = os.path.join(os.path.abspath(data_dir), 'train')  # Inside, each class should have it's own folder
    validation_dir = os.path.join(os.path.abspath(data_dir), 'val')  # each class should have it's own folder

    get_data(train_dir, validation_dir)

    # release memory
    k.clear_session()