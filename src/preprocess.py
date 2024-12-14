from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_data(x_train, y_train):
    # Data augmentation for better generalization
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)
    return datagen
