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
    
    try:
    
        filename = 'ner_model.sav'
        
        pickle.dump(model, open(out_folder+'/'+filename, 'wb'))
        
    except Exception as e: 
    
        print('** Failed to Save NER Model..'+ str(e))


def evaluate_model(model, test_data):
    
    """
    Evaluate your model against the test data.
    """
    
    try:
    
        print(classification_report(y_pred=model.predict(test_data['test_set_features']), y_true=test_data['test_set_labels'], labels=test_data['classes']))
        
    except Exception as e: 
    
        print('** Failed to Evaluate NER Model..'+ str(e))


def split_data(data):
    
    """
    Generate data splits
    
    """
    
    try:
        
        dependant_variable = list(data.columns)[1]
        
        features = data.drop(dependant_variable, axis=1)
        
        X = vectorize_dataset(features)
    
        y = data[dependant_variable].values
        
        classes = np.unique(y)
        
        classes = classes.tolist()
        
        return train_test_split(X, y, test_size = 0.33, random_state=0)
    
    except Exception as e: 
    
        print('** Failed Splitting Dataset..'+ str(e))
    
    
   



def load_data(in_folder: str):
    
    """
    
    The in_folder will contain two files:
     - person.ttl
     - name.ttl

    You will need to combine the data to generate the y values (0 or 1),
    and train the model (see readme).
    
    """
    
    try: 
    
        df = pd.read_csv (in_folder+'/ner_dataset.csv')
        
        return df
    
    except Exception as e: 
    
        print('** Failed Loading Dataset..'+ str(e))
    

    

def vectorize_dataset(features):
    
    """
    
      Vectorizing nominal features of the dataset
    
    """
    
    try: 
        
        v = DictVectorizer(sparse=False)
        
        return v.fit_transform(features.to_dict('records'))
    
    except Exception as e: 
    
        print('** Failed Vectorizing Dataset..'+ str(e))
    


def train(in_folder: str, out_folder: str) -> None:
    
    """
    
    Consume the data from the input folder to generate the model
    and serialise it to the out_folder.
    
    """
    
    dataset = load_data(in_folder)
    
    classes = np.unique(dataset[list(dataset.columns)[1]].values).tolist()
    
    print('\n## Splitting Dataset into Train & Test Set:\n')
 
    X_train, X_test, y_train, y_test = split_data(dataset)
    
    print('\nShape of Train Set: '+str(X_train.shape)+',Shape of Test Set: '+str(X_test.shape)+'\n')
    
    print('\n## Training Predictive Model:\n')
    
    ner_model = Perceptron(verbose=10, n_jobs=-1, max_iter=5)
    
    ner_model.partial_fit(X_train, y_train, classes)
    
    print('\n## Training Completed...\n')
    
    print('\n## Test Data Evaluation:\n')
    
    evaluate_model(ner_model, {'test_set_features': X_test, 'test_set_labels': y_test, 'classes': classes.copy()})
    
    print('\n## Saving NER Model...\n')
    
    save_model(ner_model,out_folder)
    
    print('\n## NER Model Saved in "'+out_folder+'" directory...\n')
    

    

    
  

if __name__ == '__main__':
  fire.Fire(train)
