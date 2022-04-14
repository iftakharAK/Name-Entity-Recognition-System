import re
import fire
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

extract_limited_data_points = True

max_number_of_extraction_data_samples = 10000

feature_extractor = DictVectorizer(sparse=False)
    

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
                
        X = extract_feature_from_dataset(features)
            
        y = data[dependant_variable].values
        
        classes = np.unique(y)
        
        return train_test_split(X, y, test_size = 0.33, random_state=0)
    
    except Exception as e: 
    
        print('** Failed Splitting Dataset..'+ str(e))
    
def extract_person_object_id(line: str):  
    
    """
    
    Extract Valid Person Id from person.ttl entry
    
    """
    
    subject_id_string = line.split(' ')[0]
    
    middle_subject_id_string = re.search('<(.*)>', subject_id_string)
    
    middle_id_string = middle_subject_id_string.group(1)
    
    list_subject_id_string = middle_id_string.split('/')
    
    return list_subject_id_string[len(list_subject_id_string)-1]


def extract_name_entity_attributes(line: str):
    
    """
    
    Extract name entity attributes from name.ttl entry

    """

    subject_id_string = line.split(' ')[0]
    
    subject_type_string = line.split(' ')[1]
    
    middle_subject_id_string = re.search('<(.*)>', subject_id_string)
    
    middle_id_string = middle_subject_id_string.group(1)
    
    list_subject_id_string = middle_id_string.split('/')
    
    subject_id = list_subject_id_string[len(list_subject_id_string)-1]
    
    middle_subject_type_string = re.search('<(.*)>', subject_type_string)
    
    middle_type_string = middle_subject_type_string.group(1)
    
    list_middle_type_string = middle_type_string.split('/')
    
    subject_type = list_middle_type_string[len(list_middle_type_string)-1]
    
    feature_val_obj = re.search('"(.*)"', line)
    
    feature_val = feature_val_obj.group(1)
    
    return subject_id , subject_type , feature_val
   



def load_data(in_folder: str):
    
    """
    
    The in_folder will contain two files:
     - person.ttl
     - name.ttl

    You will need to combine the data to generate the y values (0 or 1),
    and train the model (see readme).
    
    """
    
    names , person_ids , data_list = [] , [] , []
                    
    current_line = 0
    
    print('\n ## Extracting Target Label id from person.ttl:\n')
    
    try:
        
        with open(in_folder+'/person.ttl',encoding='utf-8') as file:
    
            for i, line in enumerate(tqdm(file)):
                                  
                    person_ids.append(extract_person_object_id(line))
                                    
                    current_line += 1
                    
                    if current_line > max_number_of_extraction_data_samples and extract_limited_data_points:
                        
                        break
                    
                   
    except Exception as e: 
                        
        print('** Failed Extracting Labels..'+ str(e))
   
             
    current_line = 0
        
    print('\n ## Constructing Dataset from <name.ttl,person.ttl> : <x,y>:\n')
     
    try:    
        
        with open(in_folder+'/name.ttl',encoding='utf-8') as file:
    
            for i, line in enumerate(tqdm(file)):
                                                                  
                    subject_id , subject_type , feature_val = extract_name_entity_attributes(line)
                    
                    if subject_type == 'name' and subject_id in person_ids:
                        
                          data_list.append([feature_val,1])
                          
                    else:
                                            
                        data_list.append([feature_val,0]) 
                    
                    current_line += 1
                    
                    if current_line > max_number_of_extraction_data_samples and extract_limited_data_points:
                        
                        break      
               
    except:
                    
        print('** Failed Dataset Construction..'+ str(e))
          
    dataset = pd.DataFrame(data_list, columns =['name_entity', 'label'])
        
    return dataset
    
 
  
    

    

def extract_feature_from_dataset(features):
    
    """
    
      Vectorizing nominal features of the dataset
    
    """
    
    try: 
                
        return feature_extractor.fit_transform(features.to_dict('records'))
    
    except Exception as e: 
    
        print('** Failed Vectorizing Dataset..'+ str(e))
        
    
def classify_new_instances(new_instance_directory,saved_model_directory):
    
    try:
    
        with open(new_instance_directory+'/novel_instance.txt') as f:
            
            lines = f.readlines()
        
        novel_set = pd.DataFrame(lines,columns =['neme_entity'])
        
        try: 
    
            novel_set = preprocess_dataset(novel_set)
            
            print('\n## Preprocessed NovelSet:\n')
            
            print(novel_set)
        
        except Exception as e: 
    
            print('** Skipping Preprocessing Step..'+ str(e))
        
        transformed_new_instances = feature_extractor.transform(novel_set.to_dict('records'))
        
        loaded_model = pickle.load(open(saved_model_directory+'/ner_model.sav', 'rb'))
            
        predicted_labels = loaded_model.predict(transformed_new_instances)
        
        lines = [entity.strip()for entity in lines]
    
        labeled_novel_set = pd.DataFrame(list(zip(lines, predicted_labels)),columns =['Name Entity', 'Predicted Label'])
        
        labeled_novel_set["Predicted Label"].replace({0: False, 1: True}, inplace=True)
        
        labeled_novel_set.to_csv('labeled-novel-instances-file/labeled_novel_set_file.csv')
        
        print('\n## Labeled Novel Instances:\n',labeled_novel_set)
        
    except Exception as e: 
    
        print('** Failed Labeling Novel Instances..'+ str(e))

            
def preprocess_dataset(dataset):
    
    import nltk
    
    from nltk.stem import WordNetLemmatizer 
    
    print('\n## Downloading NLTK Resources....\n')
    
    nltk.download('all',quiet=True)
    
    lemmatizer = WordNetLemmatizer()
       
    feature_col = list(dataset.columns)[0]
    
    dataset[feature_col] = dataset[feature_col].str.replace('[^\w\s\n]',' ')
    
    dataset[feature_col] = dataset[feature_col].str.strip()
    
    dataset[feature_col] = dataset[feature_col].str.lower()    
    
    feature_col_val = dataset[feature_col].tolist()
    
    lemitized_ftrs = [lemmatizer.lemmatize(val) for val in feature_col_val]
    
    dataset[feature_col] = lemitized_ftrs
    
    return dataset
    

     

def train(in_folder: str, out_folder: str) -> None:
    
    """
    
    Consume the data from the input folder to generate the model
    and serialise it to the out_folder.
    
    """
    
    dataset = load_data(in_folder)
    
    print(dataset,'  \n\n',dataset.describe(include='all'))
        
    try: 
    
        dataset = preprocess_dataset(dataset)
        
        print('\n## Preprocessed Dataset:\n')
        
        print(dataset,'  \n\n',dataset.describe(include='all'))
        
    except Exception as e: 
    
        print('** Skipping Preprocessing Step..'+ str(e))
    
    classes = np.unique(dataset[list(dataset.columns)[1]].values).tolist()
    
    print('\n## Splitting Dataset into Train & Test Set:\n')
 
    X_train, X_test, y_train, y_test = split_data(dataset)
    
    print('\nShape of Train Set: '+str(X_train.shape)+',Shape of Test Set: '+str(X_test.shape)+'\n')
    
    print('\n## Training Predictive Model:\n')
    
    ner_model = Perceptron(verbose=10, n_jobs=-1, max_iter=1000)
    
    ner_model.partial_fit(X_train, y_train, classes)
    
    print('\n## Training Completed...\n')
    
    print('\n## Test Data Evaluation:\n')
    
    evaluate_model(ner_model, {'test_set_features': X_test, 'test_set_labels': y_test, 'classes': classes.copy()})
    
    print('\n## Saving NER Model...\n')
    
    save_model(ner_model,out_folder)
    
    print('\n## NER Model Saved in "'+out_folder+'" directory...\n')
    
    

    

    
  

if __name__ == '__main__':
  fire.Fire(train)
