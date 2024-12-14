from model import build_model
from data_loader import load_data
from preprocess import augment_data

def train_model():
    # Load and preprocess data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()

    # Augment data (optional)
    datagen = augment_data(x_train, y_train)

    # Build the model
    model = build_model()

    # Train the model
    model.fit(datagen.flow(x_train, y_train, batch_size=64),
              epochs=10,
              validation_data=(x_val, y_val))

    # Save the model
    model.save('cifar10_model.h5')
    print("Model trained and saved.")

if __name__ == "__main__":
    train_model()
