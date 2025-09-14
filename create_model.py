import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Create a CNN model for plant disease classification
def create_plant_disease_model():
    model = models.Sequential([
        layers.Input(shape=(160, 160, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(47, activation='softmax')  # 47 classes based on your labels
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    print("Creating plant disease recognition model...")
    model = create_plant_disease_model()

    # Create some dummy training data for demonstration
    # In practice, you would train with your actual dataset
    dummy_x = np.random.random((100, 160, 160, 3))
    dummy_y = tf.keras.utils.to_categorical(np.random.randint(0, 47, 100), 47)

    print("Training model with dummy data (for demonstration)...")
    model.fit(dummy_x, dummy_y, epochs=1, batch_size=32, verbose=1)

    # Save the model
    model.save("models/plant_disease_recog_model_pwp.keras")
    print("Model saved to models/plant_disease_recog_model_pwp.keras")

    print("Model summary:")
    model.summary()