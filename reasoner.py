from ast import arg
from rdflib import Graph, RDFS, RDF, URIRef, Namespace, Literal, XSD, FOAF, OWL
from owlrl import DeductiveClosure, RDFS_Semantics
import random, sys
import sys
from tqdm import tqdm
import os
from string import Template
import re
import json
import rdflib 
import requests
from bs4 import BeautifulSoup
from pprint import pprint
from inflection import camelize
from time import sleep

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os, random, json, logging, csv
import spotlight

import pandas as pd

import torch


"""
sequence of events into an article

schema.org information consistent with dbpedia or are we missing information like location date etc.

additional resources like wikidata.

truth assesment percentages or over subgraphs
"""

def generate_wikidata_consistency_state(csv_file):

    csv = pd.read_csv(csv_file)

    dbpedia = csv["Consistency_State"]

    consistency_state = []

    for elements in dbpedia:
        if eval(elements) == [True]:
            consistency_state.append(1)
        else:
            elements = eval(elements)

            if elements.count(True) > elements.count(False):
                consistency_state.append(1)
            else:
                consistency_state.append(0)

    csv["Wikidata_Consistency_State"] = consistency_state

    csv.to_csv("factsheet.csv", sep='\t')

    return True
    

def check_for_complete_triples(csv_file):

    csv = pd.read_csv(csv_file)

    dbpedia_annotated_triples = csv["Dbpedia_Annotated_Triples"]

    consistency_state_column = []

    for i in tqdm(range(0, len(dbpedia_annotated_triples))):

        triple_set = eval(dbpedia_annotated_triples[i])

        consistency_state = []

        for triples in triple_set:

            for triple in triples:
                subject = triple[0]
                predicates = triple[1]
                objects = triple[2]

                if subject != None and predicates != None:
                    print("Subject ", subject)
                    print("Predicate ", predicates)
                    print("Object ", objects)
               
        
        consistency_state_column.append(0)

    csv["Dbpedia_Consistency_State"] = consistency_state_column

    csv.to_csv("factsheet.csv", sep='\t')

    return True

    
                

def continue_dbpedia_annotation(csv_file):

    csv = pd.read_csv("factsheet.csv")

    annotated_triples = csv['Annotated_Triples']

    dbpedia_annotated_triples_ = csv["Dbpedia_Annotated_Triples"]

    index = 0

    for i in range(0, len(dbpedia_annotated_triples_)):

        instance = dbpedia_annotated_triples_[i]

        if instance == '[]':

            index = i

            break

    dbpedia_annotated_triples_frame = dbpedia_annotated_triples_[0:index]

    dbpedia_annotated_triples = []
    for triple in dbpedia_annotated_triples_frame:
        dbpedia_annotated_triples.append(
            triple
        )

    for i in tqdm(range(index, len(annotated_triples))):

        triple_set = eval(annotated_triples[i])

        dbpedia_triple_set = []

        for triple_list in triple_set:

            dbpedia_triple_list = []

            for triple in triple_list:

                subject = find_dbpedia_iri(triple[0])
                predicate = find_dbpedia_iri(triple[1])
                object_ = find_dbpedia_iri(triple[2])

                dbpedia_triple_list.append(
                    (
                        subject,
                        predicate,
                        object_
                    )
                )

            dbpedia_triple_set.append(
                dbpedia_triple_list
            )

        dbpedia_annotated_triples.append(
            dbpedia_triple_set
        )

        list_instance = []
        for instance in annotated_triples[len(dbpedia_annotated_triples):len(annotated_triples)]:
            list_instance.append(
                []
            )

        csv["Dbpedia_Annotated_Triples"] = dbpedia_annotated_triples + list_instance

        csv.to_csv("factsheet.csv", sep='\t')

    return True
    



def find_dbpedia_iri(iri):

    if "http://" not in iri:
        return iri
    
    dbpedia_iri = None

    g = Graph()
    g.parse(iri)
    
    if "/P" in iri:
        filter_ = URIRef("http://www.wikidata.org/prop/direct/P1628")
    else:
        filter_ = URIRef("http://www.wikidata.org/prop/direct/P1709")

    for s, p, o in g.triples((URIRef(iri), filter_, None)):
        if "dbpedia" in o:
            dbpedia_iri = o

    print("dbpedia iri ", dbpedia_iri)

    return dbpedia_iri
    

    

def obtain_dbpedia_triples(csv_file):

    csv = pd.read_csv("factsheet.csv")

    annotated_triples = csv["Annotated_Triples"]

    dbpedia_annotated_triples = []

    for i  in tqdm(range(0, len(annotated_triples))):

        triple_set = eval(annotated_triples[i])

        dbpedia_triple_set = []
        
        for triple_list in triple_set:

            dbpedia_triple_list = []

            for triple in triple_list:

                subject = find_dbpedia_iri(triple[0])
                predicate = find_dbpedia_iri(triple[1])
                object_ = find_dbpedia_iri(triple[2])

                dbpedia_triple_list.append(
                    (
                        subject,
                        predicate,
                        object_
                    )
                )

            dbpedia_triple_set.append(
                dbpedia_triple_list
            )

        dbpedia_annotated_triples.append(
            dbpedia_triple_set
        )

        list_instance = []
        for instance in annotated_triples[len(dbpedia_annotated_triples):len(annotated_triples)]:
            list_instance.append(
                []
            )

        csv["Dbpedia_Annotated_Triples"] = dbpedia_annotated_triples + list_instance

        csv.to_csv("factsheet.csv", sep='\t')
                
    return True


def validate_triple_set(csv_file):

    csv = pd.read_csv("factsheet.csv")

    annotated_triples = csv["Annotated_Triples"]

    consistency_state_column = []
    
    for i in tqdm(range(0, len(annotated_triples))):

        triple_set = eval(annotated_triples[i])

        consistency_state = []
        
        for triple_list in triple_set:

            consistency_state.append(False)

            graph_objects = {}

            for triples in triple_list:

                subject = triples[0]

                if subject not in graph_objects:

                    graph_object = Graph()

                    graph_object.parse(
                        subject
                    )

                    graph_objects[subject] = graph_object

            for triple in triple_list:

                graph = graph_objects[triple[0]]

                mapping = None
                for s, p, o in graph.triples(
                        (
                            URIRef(triple[1]),
                            URIRef("http://wikiba.se/ontology#directClaim"),
                            None
                        )
                ):
                    if not mapping:
                        mapping = URIRef(o)

                if not mapping:
                    mapping = triple[1]
                    
                mapped_objects = []

                for s, p, o in graph.triples(
                        (
                            URIRef(triple[0]),
                            mapping,
                            None
                        )
                ):
                    mapped_objects.append(
                        o
                    )

                if URIRef(triple[2]) in mapped_objects or Literal(triple[2]) in mapped_objects:
                    print("Yes match", triple[2])
                    consistency_state[-1]=True

        consistency_state_column.append(
            consistency_state
        )

    csv["Consistency_State"] = consistency_state_column

    csv.to_csv("factsheet.csv", sep='\t')

    return True
    
                    
    

def extract_line_triples(line):

    triple_list = []

    for triple in line:

        subject = triple[0]
        predicate = triple[1]
        object_ = triple[2]
        
        subject_wikidata_ref = query_wikidata(subject)
        predicate_wikidata_ref = query_wikidata(predicate)
        object_reference = query_wikidata(object_)
        if len(object_reference) > 0:
            object_wikidata_ref = object_reference
        else:
            object_wikidata_ref = [object_]

        triple_set = []

        for subject_ in subject_wikidata_ref:
            for predicate_ in predicate_wikidata_ref:
                for object_ in object_wikidata_ref:
                    triple_set.append(
                        (
                            subject_,
                            predicate_,
                            object_
                        )
                    )

        triple_list.append(triple_set)

    return triple_list

        

def get_wikidata_triples(csv_file):

    csv = pd.read_csv(csv_file)

    triples = csv["Triples"]

    annotated_triples = []

    for i in tqdm(range(0, len(triples))):

        sleep(3)
        
        triple_ = triples[i]
        
        triple = eval(triple_)

        #print("The triple input is ", triple)
        
        line_triples = extract_line_triples(triple)

        #print("The line triples are ", line_triples)

        annotated_triples.append(line_triples)

    csv["Annotated_Triples"] = annotated_triples

    csv.to_csv("factsheet.csv", sep='\t')

    return True


def read_extract_triples():

    csv = pd.read_csv("factsheet.csv")

    sentences = csv["Sentence"]

    triples_list = []

    for i in tqdm(range(0, len(sentences))):

        sentence = sentences[i]

        triples = extract_triples(sentence)
                
        triples_list.append(triples)

    csv["Triples"] = triples_list
    
    csv.to_csv("factsheet.csv", sep='\t')

    return True
        

def annotate(text):
  try:
    spotlight_results = spotlight.annotate('https://api.dbpedia-spotlight.org/en/annotate',text)
    urls = []
    for r in spotlight_results:
      urls.append(r['URI'])
    return urls
  except:
      return None


def extract_triples(text):

    """
        Extract triples from text.
            - input: text
            - output: extracted triples
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large").to(device)
    gen_kwargs = {
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": 1,
    }

    model_inputs = tokenizer(
        text, max_length=256, padding=True, truncation=True, return_tensors = 'pt'
    ).to(device)

    generated_tokens = model.generate(
        model_inputs["input_ids"].to(device),
        attention_mask=model_inputs["attention_mask"].to(device),
        **gen_kwargs,
    )

    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    all_triples = []
    for idx, sentence in enumerate(decoded_preds):
        print(f'Prediction triplets sentence {idx}')
        triples = extract_triplets(sentence)

        all_triples.append(
            triples
        )

    return all_triples
   

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append([subject.strip(), relation.strip(), object_.strip()])
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append([subject.strip(), relation.strip(), object_.strip()])
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append((subject.strip(), relation.strip(), object_.strip()))
    return triplets


def reasoner(turtle_file):

    """
        Check the validity of the turtle triples.
            - input: Turtle file
            - output: (Consistency Percentage Dbpedia, Consistency Percentage Wikidata)
    """

    g = Graph()
    g.parse(turtle_file, format="turtle")

    consistency_state_dbpedia = []
    
    for subject, predicate, object_ in g.triples((None, None, None)):
        consistency_state = get_consistency_state_dbpedia(
            subject,
            predicate,
            object_
        )

        consistency_state_dbpedia.append(
            consistency_state
        )
        
    
    consistency_state_wikidata = []

    for subject, predicate, object_ in g.triples((None, None, None)):
        consistency_state = get_consistency_state_wikidata(
            subject,
            predicate,
            object_
        )

        consistency_state_wikidata.append(
            consistency_state
        )

    consistency_state_wikidata_direct = get_consistency_state_wikidata_direct(
        turtle_file
    )

    truth_percentage_dbpedia = (consistency_state_dbpedia.count(True)/(consistency_state_dbpedia.count(True) + consistency_state_dbpedia.count(False))) * 100

    truth_percentage_wikidata = (consistency_state_wikidata.count(True)/(consistency_state_wikidata.count(True) + consistency_state_wikidata.count(False))) * 100

    truth_percentage_wikidata_direct = (consistency_state_wikidata_direct.count(True)/(consistency_state_wikidata_direct.count(True) + consistency_state_wikidata_direct.count(False))) * 100

    return (
        truth_percentage_dbpedia,
        truth_percentage_wikidata,
        truth_percentage_wikidata_direct
    )


def get_consistency_state_dbpedia(subject, predicate, object_):

    """
        Obtain the consistency of the object by querying dbpedia.
            - input: (subject, predicate, object to reference)
            - output: boolean
    """

    dbpedia_graph = Graph()

    try:
        dbpedia_graph.parse(
            subject
        )

        all_objects = []
        for object__ in dbpedia_graph.objects(
                subject = subject,
                predicate = predicate,
                unique = True
        ):
            all_objects.append(
                object__
            )

        if object_ in all_objects:
            return True
        else:
            print("Inconsistency ", object_, all_objects)
            return False
    
        return True
    except:
        print("URI failed: ", subject)
        return False

    
def get_consistency_state_wikidata(subject, predicate, object_):

    """
        Obtain the consistency of the object by querying dbpedia.
            - input: (subject, predicate, object to reference)
            - output: boolean
    """

    # Find mapping from dbpedia
    dbpedia_graph_subject = Graph()
    try:
        dbpedia_graph_subject.parse(
            subject
        )
    except:
        print("URI failed wikidata subject parse: ", subject)
        return False

    wikidata_ref_subject = []
    for objects in dbpedia_graph_subject.objects(
            subject,
            OWL.sameAs,
            unique=True
    ):
        if "wikidata" in objects:
            wikidata_ref_subject.append(
                objects
            )

    dbpedia_graph_predicate = Graph()
    try:
        dbpedia_graph_predicate.parse(
            predicate
        )
    except:
        print("URI failed wikidata predicate parse: ", predicate)
        return False

    wikidata_ref_predicate = []
    for objects in dbpedia_graph_predicate.objects(
            predicate,
            OWL.equivalentProperty,
            None
    ):
        if "wikidata" in objects:
            wikidata_ref_predicate.append(
                objects
            )


    dbpedia_graph_object = Graph()
    #print("The object is")
    #print(object_)
    
    wikidata_ref_object = []

    if "http://" not in object_:
        wikidata_ref_object.append(object_)
    else:
        try:
            dbpedia_graph_object.parse(
                   object_
            )
        except:
            print("URI failed wikidata object parse: ", object_)
            return False
    
        for objects in dbpedia_graph_object.objects(
            object_,
            OWL.sameAs,
                None
        ):
            if "wikidata" in objects:
                wikidata_ref_object.append(
                    objects
            )

    print("The subject predicate and object is")
    print(wikidata_ref_subject)
    print(wikidata_ref_predicate)
    print(wikidata_ref_object)

    wikidata_graph = Graph()
    wikidata_graph.parse(
        wikidata_ref_subject[0]
    )

    if len(wikidata_ref_predicate) == 0:
        print("No predicate found for ", predicate)
        return False
    else:
        mapping = None
        for s, p, o in wikidata_graph.triples(
                (
                    URIRef(wikidata_ref_predicate[0]),
                    URIRef("http://wikiba.se/ontology#directClaim"),
                    None
                )
        ):
            if not mapping:
                mapping = URIRef(o)
            
    all_objects = []
    for s, p, o in wikidata_graph.triples(
            (
                URIRef(wikidata_ref_subject[0]),
                mapping,
                None
            )
    ):
        all_objects.append(
            o
        )

    consistency_found = False
    for object__ in all_objects:
        if len(wikidata_ref_object) > 0 and wikidata_ref_object[0] in object__:
            consistency_found = True
            return True
        else:
            print("No object found")
            print(object__)

    if not consistency_found:
        print("Inconsistency (dbpedia instance, property, wikidata ref, all objects found)", object_, mapping, wikidata_ref_object, all_objects)
        return False
    else:
        return True


def query_wikidata(string):

    """
       String query wikidata
           - input; string to query
           - output: wikidata code
    """

    url = 'https://query.wikidata.org/sparql'
    query = """
        SELECT DISTINCT ?item
            WHERE {

               # make input string into a language-tagged string
               BIND( STRLANG("%s", "en") AS ?label ) .

               # search all items that have this languaged-tagged string as label
               ?item rdfs:label ?label .

         }
    """%(string)

    r = requests.get(url, params = {'format': 'json', 'query': query})

    try:
        data = r.json()
    
        bindings = data["results"]["bindings"]
    except:
        bindings = []
        
    results = []
    for r in bindings:
        results.append(
            r['item']["value"]
        )

    return results

def query_wikidata_property(string):

    """
       String query wikidata property wdt:P1687
           - input; string to query
           - output: wikidata code
    """
    
    url = 'https://query.wikidata.org/sparql'
    query = """
        SELECT distinct ?item ?property WHERE {  
            ?item ?label "%s"@en.
            ?item wdt:P1687 ?property . 
            ?article schema:about ?item .
            ?article schema:inLanguage "en" .
            ?article schema:isPartOf <https://en.wikipedia.org/>.	
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }    
        }
    """%(string)

    r = requests.get(url, params = {'format': 'json', 'query': query})

    try:
        data = r.json()
        
        bindings = data["results"]["bindings"]
    except:
        bindings = []
        
    results = []
    for r in bindings:
        results.append(
            r['property']["value"]
        )
        
    return results


def camel_to_sentence(string):
    """
        Input - Sting: birthPlace
        Output - Sting: birth place
    """

    # Check for camel casing
    if camelize(string, False) == string:
        new_string = ""
        for i in string:
            if i.isupper() == True:
                new_string += " "+i.lower()
            else:
                new_string += i
    else:
        new_string = string    

    return new_string


def get_consistency_state_wikidata_direct(graph_path):

    """
       Obtain the accuracy of a wikidata graph.
            - graph_path: path to dbpedia graph
            - output: list of True or False.
    """

    triple_set = obtain_wikidata_codes(graph_path)

    print("The wikidata codes are")
    print(triple_set)


    consistency_state = []
    
    for each_set in triple_set:

        consistency_state.append(False)
        
        graph_objects = {}
        for triple in each_set:
            subject = triple[0]
            if subject not in graph_objects:
                graph_object = Graph()
                graph_object.parse(
                    subject
                )
                graph_objects[subject] = graph_object

        for triple in each_set:
            graph = graph_objects[triple[0]]

            mapping = None
            for s, p, o in graph.triples(
                (
                    URIRef(triple[1]),
                    URIRef("http://wikiba.se/ontology#directClaim"),
                    None
                )
            ):
                if not mapping:
                    mapping = URIRef(o)

            if not mapping:
                mapping = triple[1]

            mapped_objects = []

            print("The subject is ", triple[0])
            print("The predicate is ", mapping)
            for s, p, o in graph.triples(
                    (
                        URIRef(triple[0]),
                        mapping,
                        None
                    )
            ):
                mapped_objects.append(
                    o
                )
            print("Object for comparison ", triple[2])
            print("The mapped objects are ", mapped_objects)
            if URIRef(triple[2]) in mapped_objects or Literal(triple[2]) in mapped_objects:
                print("Yes match", triple[2])
                consistency_state[-1]=True

    print("The consistency state is ", consistency_state)

    return consistency_state
                    
            
def obtain_wikidata_codes(graph_path):

    """
      Obtain the wikidata codes by sparql querying.
        - input: dbpedia graph
        - output: wikidata triple set. ( (set of triples to match, set of triples to match) )
    """

    g = Graph()

    g.parse(graph_path)

    wikidata_codes_subjects = []
    wikidata_codes_predicates = []
    wikidata_codes_objects = []

    subjects = []
    predicates = []
    objects = []

    for s, p, o in g.triples((None, None, None)):
        subjects.append(
            camel_to_sentence(
                s.split("/")[-1].replace("_", " ")
            )
        )
        predicates.append(
            camel_to_sentence(
                p.split("/")[-1].replace("_", " ")
            )
        )
        if "http:" not in o:
            objects.append(
                (
                    o,
                    "Literal"
                )
            )
        else:
            objects.append(
                (
                    camel_to_sentence(
                        o.split("/")[-1].replace("_", " ")
                    ),
                    "IRI"
                )
            )

    triples = []
    for i in range(0, len(subjects)):
        triples.append(
            (
                subjects[i],
                predicates[i],
                objects[i]
            )
        )
        
    print("---------------")
    print("The triples are")
    print(triples)
        
    new_triples = []
    for triple in triples:
        sleep(5)
        new_triples.append(
            generate_wikidata_triples(
                triple
            )
        )
            
    return new_triples


def generate_wikidata_triples(triple):

    """
        Generating wikidata triples
          - input: triple
          - output: wikidata triple set.
    """

    subject = triple[0]
    predicate = triple[1]
    object_ = triple[2]

    wikidata_subject = query_wikidata(
        subject
    )

    wikidata_predicate = query_wikidata_property(
        predicate
    )

    wikidata_object = []
    if object_[1] == "Literal":
        wikidata_object = [object_[0]]
    else:
        wikidata_object = query_wikidata(
            object_[0]
        )

    triple_set = []

    for x in wikidata_subject:
        for y in wikidata_predicate:
            for z in wikidata_object:
                triple_set.append(
                    (
                        x, y, z
                    )
                )
    
    return triple_set


"""
SELECT ?b ?bLabel
WHERE
{
  ?item rdfs:label "point in time"@en.
  ?item ?a ?b.

  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } 
  ?prop wikibase:directClaim ?a .
}
"""
