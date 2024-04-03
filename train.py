import json
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from keras.preprocessing.image import ImageDataGenerator

async def train_new(layers : int = 64):
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
    
    classes = val_data.class_indices.keys()
    
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

    callback_acc = EarlyStopping(monitor='val_accuracy', patience=10)
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=100,
        callbacks=[callback_acc])
    
    result = {}

    model.save(model_path+'/temp/SavedModel.h5')
    with open('/temp/species.json', 'w') as json_file:
        json.dump(classes, json_file)

    result['best_val_accuracy'] = max(history.history['val_accuracy'])
    result['best_val_loss'] = min(history.history['val_loss'])
    result['best_accuracy'] = max(history.history['accuracy'])
    result['best_loss'] = min(history.history['loss'])

    return result

    


    
