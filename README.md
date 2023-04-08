# krw_project

### Instructions to run the project

Our project is based on a csv file. The various functions in the ```reasoner.py``` file contribute to every column. To generate the various columns, we ran the following functions. 


1. We added all the sentences from ChatGPT to the CSV file. (column: Sentence)

2. To extract triples using the rebel api, we ran the function below. (column: Triples)

```
	read_extract_triples()
```

3. Obtain triples from Wikidata. (column: Annotated_Triples)

```
	get_wikidata_triples("factsheet.csv")
```

4. Validating Wikidata triples. (column: Consistency_State)

```
	validate_triple_set("factsheet.csv")
```

5. Annotate the Wikidata triples to DBpedia format. (column: DBpedia_Annotated_Triples)

```
	continue_dbpedia_annotation("factsheet.csv")
```

6. Check for the consistency of DBpedia triples. We observed that no complete triples were returned from step 5 so all the consistency state values here were false. (column: Dbpedia_Consistency_State)

```
	check_for_complete_triples("factsheet.csv")
```

7. We now have the consistency state of the wikidata triples, the dbpedia triples and the manually checked triples to verify how accurate the predictions were. 
