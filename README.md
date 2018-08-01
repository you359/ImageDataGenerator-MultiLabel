# ImageDataGenerator
keras ImageDataGenerator for Multi Labeled Data

Keras ImageDataGenerator is useful API for loading train/val/test dataset from data array or file system
it also provide data augmentation method.

However unfortunatly Keras ImageDataGenerator not yet support Multi Labeled Dataset.

meaning of Multi Labeled Dataset is that one data has multiple labels.

<img src='https://www.google.co.kr/url?sa=i&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwios7G9ksvcAhUNhrwKHVIBBzoQjRx6BAgBEAU&url=https%3A%2F%2Fpixabay.com%2Fen%2Ffire-smoke-sunset-red-disaster-1265716%2F&psig=AOvVaw3OBucSiEn7rI-Gd4fQcnsU&ust=1533188470008456'>
for example, above image data have two labels which was fire and smoke.

for supporting Multi Labeled Dataset on Image Data Generator, i modified some part of keras.preprocessing.image

## How to use?
import utils to your ython script

```python
from utils import *
```

this utils.py file almost same with keras.preprocessing.image without some function

make data file structure like bellow(Fire-Smoke forlder have multi labeled images)

```
├── Fire(folder)
│   └── XXX-1.jpeg
│   └── ...
├── Smoke(folder)
│   └── XXX-1.jpeg
│   └── ...
├── Fire-Smoke(folder)
│   └── XXX-1.jpeg
│   └── ...hav
```

then, setting parameter class_mode to "multi_categorical" in function "flow_from_directory"

if subdirectory have some seperator("-"), this utils will automatically identify that this subdirectory is multi labeled class and provide multi label to this class 

## Test ImageDataGenerator for Multi Labeled Dataset
first modify igd_test.py
```
if __name__ == '__main__':
    data_dir = 'path/to/dataset' # modify to your dataset path
    train_dir = os.path.join(os.path.abspath(data_dir), 'train')  # Inside, each class should have it's own folder
    validation_dir = os.path.join(os.path.abspath(data_dir), 'val')  # each class should have it's own folder

    get_data(train_dir, validation_dir)

    # release memory
    k.clear_session()
```

and running below command

```shell
python idg_test.py
```

then, some number of train/val data and labels are printed on your terminals

## Reference
[1] [keras.io](https://keras.io/preprocessing/image/) <br/>
[2] [https://github.com/keras-team/keras-preprocessing](https://github.com/keras-team/keras-preprocessing)