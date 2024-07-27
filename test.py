import numpy as np
import json
from sklearn.metrics import precision_score, recall_score
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

def test(model_name : str)->dict[str,any]: 
    dataset_path = './../dataset/training'
    model_path = './../models'

    test_datagen = ImageDataGenerator(
            rescale=1/255.,
        )

    IMG_SIZE = (224, 224)
    test_dir = dataset_path + '/testing'
    test_data = test_datagen.flow_from_directory(test_dir, shuffle=False,
                                                    target_size=IMG_SIZE,
                                                    batch_size=8)

    model = load_model(model_path + '/' + model_name +'/SavedModel.h5')
    # with open(model_path + '/' + model_name +'/species.json',"r") as f:
    #         species = json.load(f)

    # Calculate predictions
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate precision and recall for each class
    precision_values = precision_score(test_data.classes, predicted_labels, average=None)
    recall_values = recall_score(test_data.classes, predicted_labels, average=None)
    overall_precision = precision_score(test_data.classes, predicted_labels, average='weighted')
    overall_recall = recall_score(test_data.classes, predicted_labels, average='weighted')

    class_labels = list(test_data.class_indices.keys()) 
    result = {"precision":overall_precision,
              "recall":overall_recall,
              "class":{}}
    for i,v in enumerate(class_labels):
          result["class"][v] = {"precision":precision_values[i],
                       "recall":recall_values[i]}
    return result