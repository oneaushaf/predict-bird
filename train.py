import json
import service
import numpy as np
from keras.models import load_model  # type: ignore
from keras import Sequential # type: ignore
from keras.layers import Dense, Flatten # type: ignore
from keras.callbacks import EarlyStopping # type: ignore
from keras.preprocessing.image import ImageDataGenerator
import time

is_training = False

async def train_new(layers : int = 64, callback_patience : int =10,epochs=100):
    try:
        dataset_path = './../dataset/trained'
        model_path = './../models'

        train_dir = dataset_path + '/train'
        val_dir = dataset_path + '/validation'

        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.2,
            zoom_range=0.2,
            rescale=1/255.,
            width_shift_range=0.2, 
            height_shift_range=0.2,
            brightness_range=[0.3,0.7]
        )
        val_datagen = ImageDataGenerator(
            rescale=1/255.,
        )

        IMG_SIZE = (224, 224)
        train_data = train_datagen.flow_from_directory(train_dir,
                                                    shuffle=True,
                                                    target_size=IMG_SIZE)

        val_data = val_datagen.flow_from_directory(val_dir, shuffle=False,
                                                    target_size=IMG_SIZE)
        
        classes = list(val_data.class_indices.keys())
        
        base_model = load_model(model_path+'/base/SavedModel.h5')
        base_model.trainable = False

        model = Sequential([
                            base_model, # This is the MobilenetV2
                            Flatten(), # Flatten the multidimension features into 1D
                            
                            # The Fully Connected (Hidden) Layer, you may customize the number, I used 64 as it performs good enough already
                            # This can contain more layers if the performance isn't good enough
                            Dense(layers, activation='relu'),

                            # The number of units here is for the number of classes in the dataset
                            Dense(len(classes), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        callback_acc = EarlyStopping(monitor='val_accuracy', patience=callback_patience)

        timer = round(time.time())

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[callback_acc])

        model.save(model_path+'/temp/SavedModel.h5')
        with open(model_path+'/temp/species.json', 'w') as json_file:
            json.dump(classes, json_file)
        timer = round(time.time()) - timer

        result = {
            "success" : True,
            "data":{
                "best_accuracy" : max(history.history['accuracy']),
                "best_loss" : max(history.history['loss']),
                "best_val_accuracy" : max(history.history['val_accuracy']),
                "best_val_loss" : max(history.history['val_loss']),
                "training_time" : timer,
                "epochs":len(history.history['loss'])
            }
        }
    except Exception as e : 
        global is_training
        is_training = False
        result = {
            "success" : False ,
            "message" : "failed to train",
            "error" : str(e)
        }
    service.make_request("http://127.0.0.1:3000/models/train/done",result)

async def train_based(callback_patience : int=10,base_model : str="latest", epochs : int = 100):
    global is_training
    try :
        dataset_path = './../dataset/trained'
        model_path = './../models'

        train_dir = dataset_path + '/train'
        val_dir = dataset_path + '/validation'

        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=0.2,
            zoom_range=0.2,
            rescale=1/255.,
            width_shift_range=0.2, 
            height_shift_range=0.2,
            brightness_range=[0.3,0.7]
        )
        val_datagen = ImageDataGenerator(
            rescale=1/255.,
        )

        IMG_SIZE = (224, 224)
        train_data = train_datagen.flow_from_directory(train_dir,
                                                    shuffle=True,
                                                    target_size=IMG_SIZE)

        val_data = val_datagen.flow_from_directory(val_dir, shuffle=False,
                                                    target_size=IMG_SIZE)
        
        classes = list(val_data.class_indices.keys())
        
        model = load_model(model_path + '/' + base_model +'/SavedModel.h5')

        for layer in model.layers[:-2]: 
            layer.trainable = False

        weights = model.layers[-1].get_weights()
        shape = weights[0].shape[0]
        weights[1] = np.concatenate((weights[1], np.zeros(1)), axis=0)
        weights[0] = np.concatenate((weights[0], -0.0001 * np.random.random_sample((shape, 1)) + 0.0001), axis=1)

        model.pop() 
        model.add(Dense(11,activation="softmax",name="output"))
        model.get_layer("output").set_weights(weights)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        callback_acc = EarlyStopping(monitor='val_accuracy', patience=callback_patience)

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=[callback_acc])

        timer = round(time.time())
        model.save(model_path+'/temp/SavedModel.h5')
        with open(model_path+'/temp/species.json', 'w') as json_file:
            json.dump(classes, json_file)
        timer = round(time.time()) - timer

        is_training = False

        result = {
            "success" : True,
            "message" : "data trained succesfully",
            "error":"",
            "data":{
                "best_accuracy" : max(history.history['accuracy']),
                "best_loss" : max(history.history['loss']),
                "best_val_accuracy" : max(history.history['val_accuracy']),
                "best_val_loss" : max(history.history['val_loss']),
                "training_time" : timer,
                "epochs":len(history.history['loss'])
            }
        }
    except Exception as e : 
        is_training = False
        result = {
            "success" : False ,
            "message" : "failed to train",
            "error" : str(e),
            "data":""
        }
    service.make_request("http://127.0.0.1:3000/models/train/done",result)


    




    


    
