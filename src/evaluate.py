from tensorflow.keras.models import load_model
from data_loader import load_data

def evaluate_model():
    # Load the trained model
    model = load_model('cifar10_model.h5')

    # Load the test data
    _, _, _, _, x_test, y_test = load_data()

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc}")

if __name__ == "__main__":
    evaluate_model()
