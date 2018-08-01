import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import l2
from keras.models import Model
from utils import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as k


# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# hyper parameters for model
nb_classes = 3  # number of classes
img_width, img_height = 299, 299  # change based on the shape/structure of your images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 200  # number of iteration the algorithm gets trained.

# learn_rate = 1e-4  # sgd learning rate
# momentum = .9  # sgd momentum to avoid local minimum


def train(train_data_dir, validation_data_dir, model_path):
    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
    # base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
    base_model = InceptionV3(input_shape=(img_width, img_height, 3), include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = layers.Dense(nb_classes, activation='sigmoid')(x)

    # add your top layer block to your base model
    model = Model(base_model.input, predictions)
    print(model.summary())

    for layer in model.layers:
        layer.trainable = True
        layer.kernel_regularizer = l2(0.05)

    model.compile(optimizer='nadam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='multi_categorical')

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='multi_categorical')

    # save weights of best training epoch: monitor either val_loss or val_acc
    final_acc_weights_path = os.path.join(os.path.abspath(model_path), 'model_acc_weights.h5')
    final_loss_weights_path = os.path.join(os.path.abspath(model_path), 'model_loss_weights.h5')

    callbacks_list = [
        ModelCheckpoint(final_acc_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        ModelCheckpoint(final_loss_weights_path, monitor='val_loss', verbose=1, save_best_only=True),
        # EarlyStopping(monitor='val_loss', patience=15, verbose=0),
        TensorBoard(log_dir='graph/train', histogram_freq=0, write_graph=True)
    ]

    # fine-tune the model
    model.fit_generator(train_generator,
                        epochs=nb_epoch,
                        validation_data=validation_generator,
                        callbacks=callbacks_list)

    # save model
    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
        json_file.write(model_json)


if __name__ == '__main__':
    data_dir = 'path/to/dataset'
    train_dir = os.path.join(os.path.abspath(data_dir), 'train')  # Inside, each class should have it's own folder
    validation_dir = os.path.join(os.path.abspath(data_dir), 'val')  # each class should have it's own folder
    model_dir = 'model/'

    os.makedirs(model_dir, exist_ok=True)

    train(train_dir, validation_dir, model_dir)  # train model

    # release memory
    k.clear_session()