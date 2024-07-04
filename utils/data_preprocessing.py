from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def get_data_generator(data_dir,target_size=(128,128),batch_size = 32,augment=True):
    if augment:
        datagen = ImageDataGenerator(
            rescale = 1.0/255,
            rotation_range = 60,
            width_shift_range = 0.4,
            height_shift_range = 0.4,
            shear_range = 0.4,
            zoom_range = 0.4,
            horizontal_flip = True,
            fill_mode = 'nearest'
        )
    else:
        datagen = ImageDataGenerator(
            rescale = 1.0/255
        )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size = target_size,
        batch_size = batch_size,
        color_mode = 'grayscale',
        class_mode = 'categorical'
    )
    return train_generator  
def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale if needed
    image = image.resize((128, 128))  # Resize to match the input size of your model
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension if needed
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
