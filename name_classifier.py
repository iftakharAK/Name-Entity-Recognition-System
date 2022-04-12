import fire

import pickle

import numpy as np

import pandas as pd

from sklearn.linear_model import Perceptron

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split



from sklearn.feature_extraction import DictVectorizer




    

def save_model(model, out_folder: str):
    
    """
    Serialise the model to an output folder 
    """
    
    filename = 'ner_model.sav'
    
    pickle.dump(model, open(out_folder+'/'+filename, 'wb'))


def evaluate_model(model, test_data):
    """
    Evaluate your model against the test data.
    """
    print('\n## Test Data Evaluation:\n')
    print(classification_report(y_pred=model.predict(test_data['test_set_features']), y_true=test_data['test_set_labels'], labels=test_data['classes']))


def split_data(data):
    """
    Generate data splits
    """
    
    #dataset_columns = list(data.columns)
    
    dependant_variable = list(data.columns)[1]
    
    features = data.drop(dependant_variable, axis=1)
    
    X = vectorize_dataset(features)

    y = data[dependant_variable].values
    
    classes = np.unique(y)
    
    classes = classes.tolist()
    
    return train_test_split(X, y, test_size = 0.33, random_state=0)
    
    
   



def load_data(in_folder: str):
    
    """
    The in_folder will contain two files:
     - person.ttl
     - name.ttl

    You will need to combine the data to generate the y values (0 or 1),
    and train the model (see readme).
    """
    
    df = pd.read_csv ('in-folder/ner_dataset.csv')
    

    return df

def vectorize_dataset(features):
    
    v = DictVectorizer(sparse=False)
    
    return v.fit_transform(features.to_dict('records'))
    


def train(in_folder: str, out_folder: str) -> None:
    
    """
    
    Consume the data from the input folder to generate the model
    and serialise it to the out_folder.
    
    """

    print(in_folder,'  ',out_folder)
    
    dataset = load_data(in_folder)
    
    classes = np.unique(dataset[list(dataset.columns)[1]].values).tolist()
    
    X_train, X_test, y_train, y_test = split_data(dataset)
    
    ner_model = Perceptron(verbose=10, n_jobs=-1, max_iter=5)
    
    ner_model.partial_fit(X_train, y_train, classes)
    
    evaluate_model(ner_model, {'test_set_features': X_test, 'test_set_labels': y_test, 'classes': classes.copy()})
    
    save_model(ner_model,out_folder)
    
    print(X_train,X_train.shape)
    
    print(X_test,X_test.shape)
    
    print(y_train,y_train.shape)
    
    print(y_test,y_test.shape)
    
    print(classes)

    

    

    pass
  

if __name__ == '__main__':
  fire.Fire(train)
