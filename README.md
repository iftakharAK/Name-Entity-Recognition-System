### People Name Classifier


Your task is to submit code that allows us to determine whether a given string is the name of a person or not. Here are some expectations: 

* “John Johnson” -> True 
* “Kahfsgjgfjsdgjhfgdjs” -> False 
* “Book” -> False 
* “Jean Claude von Muchausen Gordon-Smith” -> True 

In order to do this, we prepared a toy dataset provided by the DBpedia Association at http://wiki.dbpedia.org/downloads-2016-10#p10608-2. The datasets contain RDF triples that represent facts about the world in the form: subject - predicate - object (e.g. “John loves Mary”). We prepared the dataset along the following way:

1. We extracted the names everything in the knowledge base from the [mappings-based literals file](http://downloads.dbpedia.org/2016-10/core-i18n/en/mappingbased_literals_wkd_uris_en.ttl.bz2) (meaning that the object of the triple is a string literal) and stored into a file (`grep 'foaf/0.1/name' mappingbased_literals_wkd_uris_en.ttl > name.ttl`)

2. We extracted all the entities that are people from the [transitive-type file](http://downloads.dbpedia.org/2016-10/core-i18n/en/instance_types_transitive_wkd_uris_en.ttl.bz2) (meaning the object of the tripple is a type, for example `Person`) and stored into a file (`grep 'ontology/Person' instance_types_transitive_wkd_uris_en.ttl > person.ttl`) 

Therefore the `name.ttl` dataset would contain lines, such as:

```
<http://wikidata.dbpedia.org/resource/Q15703184> <http://xmlns.com/foaf/0.1/name> "19th National Television Awards"@en .
<http://wikidata.dbpedia.org/resource/Q15703194> <http://xmlns.com/foaf/0.1/name> "José Miguel Ruiz Cortés"@en .
<http://wikidata.dbpedia.org/resource/Q15703194> <http://xmlns.com/foaf/0.1/name> "José Ruiz"@en .
<http://wikidata.dbpedia.org/resource/Q15703238> <http://xmlns.com/foaf/0.1/name> "Rainbow Rowell"@en .
<http://wikidata.dbpedia.org/resource/Q15703254> <http://xmlns.com/foaf/0.1/name> "Escape Velocity"@en .
<http://wikidata.dbpedia.org/resource/Q15703257> <http://xmlns.com/foaf/0.1/name> "The Oath"@en .
<http://wikidata.dbpedia.org/resource/Q15703260> <http://xmlns.com/foaf/0.1/name> "Islanded in a Stream of Stars"@en .
<http://wikidata.dbpedia.org/resource/Q15703263> <http://xmlns.com/foaf/0.1/name> "Occupation"@en .
<http://wikidata.dbpedia.org/resource/Q15703266> <http://xmlns.com/foaf/0.1/name> "Precipice"@en .
<http://wikidata.dbpedia.org/resource/Q15703272> <http://xmlns.com/foaf/0.1/name> "Collaborators"@en .
<http://wikidata.dbpedia.org/resource/Q15703276> <http://xmlns.com/foaf/0.1/name> "A Measure of Salvation"@en .
<http://wikidata.dbpedia.org/resource/Q15703279> <http://xmlns.com/foaf/0.1/name> "Hero"@en .
<http://wikidata.dbpedia.org/resource/Q15703281> <http://xmlns.com/foaf/0.1/name> "Unfinished Business"@en .
<http://wikidata.dbpedia.org/resource/Q15703286> <http://xmlns.com/foaf/0.1/name> "The Passage"@en .
<http://wikidata.dbpedia.org/resource/Q15703291> <http://xmlns.com/foaf/0.1/name> "Rapture"@en .
<http://wikidata.dbpedia.org/resource/Q15703294> <http://xmlns.com/foaf/0.1/name> "The Woman King"@en .
<http://wikidata.dbpedia.org/resource/Q15703298> <http://xmlns.com/foaf/0.1/name> "A Day in the Life"@en .

```

and the `person.ttl` dataset would contain lines, like

```
<http://wikidata.dbpedia.org/resource/Q1000005> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000051> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q100005> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000061> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000085> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000203> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000235> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q10002689> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000296> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q100030> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
```

With these two files you are able to generate a dataset containing tuples of the form `(<name>, <class>)` where `<class>` would be either `1` if the subject in the `names.ttl` file is present in the `person.ttl` file or `0` otherwise (binary classification problem).

We have defined a helper `name_classifier.py` file that would help you. It has some boiler plate defined but you need to fill out yourself and write some extra functions or two. You can also solve this using another programming language like Scala , the python code might help you by providing some pseudo-code.

If you use extra libraries, please amend the provided `requirements.txt` file and this readme with instructions. Once everything is ready, we could use the tool by running:

`python name_classifier.py --in-folder <path-to-data> --out-folder <path-to-model-destination>` , where
	* `<path-to-data>` corresponds to the data containing the csv files
	* `<path-to-model-destination>` corresponds to a folder where the trained model will be serialised to


Your code in `name_classifier.py` should:
* generate a training / eval / test split (and do any necessary data pre-processing)
* train a model
* print evaluation metrics
* save it to a destination

Once you are done, submit a pull request for evaluation. You can download the dataset from [this link](https://episerver99-my.sharepoint.com/:u:/g/personal/zsolt_pocsaji_episerver_com/EeHdaCtLJP1HlurrLtBOKxcBsq4JhkiLJqw7f1sAmwa_ZQ?e=uNbuqJ).
