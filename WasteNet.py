import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import time

class WasteNet():
    """
    Holds all logic relating to the underlying model that performs image prediction.

    Author:
        Keith Hendricks
        Skyler Looney
    """

    BATCH_SIZE = 128 # Determines the batch size (Higher means faster times but higher strain on the computer)

    @classmethod
    def create_model(cls) :
        """
        Creates and outputs a new model instance.

        This function was used to initialize the model during the first run.
        It is no longer actively used, as the model has already been created
        and is loaded from storage in subsequent runs.

        """
        train = ImageDataGenerator(rescale= 1/255)
        validation = ImageDataGenerator(rescale = 1/255)

        train_dataset= train.flow_from_directory('archive/DATASET/TRAIN',
                                                target_size= (200, 200),
                                                batch_size = cls.BATCH_SIZE,
                                                class_mode = 'categorical')

        test_dataset = validation.flow_from_directory('archive/DATASET/TEST',
                                                    target_size= (200, 200),
                                                    batch_size = cls.BATCH_SIZE, 
                                                    class_mode = 'categorical')


        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax') #3 is the number of classes the output can be
        ])

        # Print the model summary
        model.summary()


        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])



        #Train the model
        history = model.fit(train_dataset, steps_per_epoch=10, epochs=75, validation_data=test_dataset) #Achieves ~90% everytime
        #history = model.fit(train_dataset, steps_per_epoch=10, epochs=10, validation_data=test_dataset) #For Quick Tests

        model.save("model/WasteNet.keras")

        loss, accuracy = model.evaluate(test_dataset, batch_size = cls.BATCH_SIZE)
        print(f"Test loss: {loss : .4f}")
        print(f"Test accuracy: {accuracy : .4f}")

        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


    @classmethod
    def predict(cls, new_image_path):
        """
        Predicts the waste category of a given image.

        This function takes the file path of an image as input, processes it through
        the trained model, and predicts whether the waste is 'organic', 'recyclable',
        or 'non-organic'. If the model is uncertain, it defaults to 'non-organic'.

        The function prints the predicted category and the time taken to make the
        prediction. It also returns the predicted category as a string.

        Parameters:
            new_image_path (str): The file path to the image to be classified.

        Returns:
            str: The predicted category of the waste.
        """

        start_time = time.perf_counter()

        loaded_model = tf.compat.v1.keras.models.load_model("model/WasteNet.keras")

        middle_time = time.perf_counter()

        img = image.load_img(new_image_path, target_size=(200, 200))
        
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis = 0)

        # Prediction
        predictions = loaded_model.predict(X)

        # predicted label
        pred_label = tf.argmax(predictions, axis =1)
        pred_label.numpy()

        determined_category = str()
        if ((1)==(predictions[0][0]) and (0)==(predictions[0][1]) and (0)==(predictions[0][2])):
            determined_category = "Non-Organic"
            print(determined_category)
        elif (1==(predictions[0][1]) and (0)==(predictions[0][0]) and (0)==(predictions[0][2])):
            determined_category = "Organic"
            print(determined_category)
        elif (1==(predictions[0][2]) and (0)==(predictions[0][1]) and (0)==(predictions[0][0])):
            determined_category = "Recyclable"
            print(determined_category)
        else :
            determined_category = "Non-Organic"
            print(determined_category)
        
        end_time = time.perf_counter()
        model_execution_time = middle_time - start_time
        function_execution_time = end_time - start_time
        print(f"Model Execution time: {model_execution_time : .4f} seconds") #Prints time to load model
        print(f"Function Execution time: {function_execution_time : .4f} seconds") #Prints time to finish running function

        return determined_category
