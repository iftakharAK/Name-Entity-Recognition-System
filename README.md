# Dependencies: #

- python ==3.16.3
- fire==0.4.0
- importlib-resources==5.4.0
- joblib==1.1.0
- numpy==1.19.5
- pandas==1.1.5
- python-crfsuite==0.9.8
- scikit-learn==0.24.2
- scipy==1.5.4
- sklearn-crfsuite==0.3.6
- tabulate==0.8.9
- termcolor==1.1.0
- threadpoolctl==3.1.0
- tqdm==4.64.0
- nltk==3.6.7
- pickle

# Installation: #
  
- Install all the Dependecies using the command below:
  
    - pip install -r requirements.txt
	
- If Using Anaconda can create the environment using the environment.yml file withthe command below :
  
    - conda env create -f environment.yml
	
- or we can manually isntall all the packages with below mentioned commands:
  
    - pip install -U scikit-learn

	- pip install numpy

	- pip install pandas

	- pip install sklearn-crfsuite

	- pip install fire
	
	- pip install pickle
	
	
# Training the model: #

- Place the "name.ttl" & "Person.ttl" FIle inside the in-folder directory.
-  TO use the tool run "python name_classifier.py --in-folder <path-to-data> --out-folder <path-to-model-destination>" , where
	- `<path-to-data>` corresponds to the data containing the .tts files
	- `<path-to-model-destination>` corresponds to a folder where the trained model will be serialised to
	- Example: "python name_classifier.py in-folder out-folder"
	
	
# Hyperparameter tunning: #

- since the size of the dataset is big we can limit the size of the constructed dataset setting  the "extract_limited_data_points" param to True and Flase otherwise.
 
  -and set the size of the dataset using "max_number_of_extraction_data_samples" param which is default 10000.
  
  - Both of the param can be found at the top of the "name_classifier.py" file.
  

- Rest of the model hyper-params can be changed and can be tuned from within the "train(in_folder: str, out_folder: str)" function.

# Novel Entity Classification: #

- To lable new instances place the new entity string within the "novel_instance.txt" file seperated by new line(\n) inside the "novel-instance-folder" Directory.

- After placing the entity's in the "novel_instance.txt" file run the "classify_new_instances(new_instance_directory,saved_model_directory)" function inside the "name_classifier.py" file where:

  - "new_instance_directory" is the folder path where "novel_instance.txt" file is placed.
  
  - "saved_model_directory" is the folder where serialised the Name Entity Recognition saved model file is stored.
  
- The predicted output of the novel entity instances will be saved in the "labeled-novel-instances-file" folders "labeled_novel_set_file.csv" file which will in the following format:

  - Input Entity , Predicted Label: {True\False}