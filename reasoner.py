from ast import arg
from rdflib import Graph, RDFS, RDF, URIRef, Namespace, Literal, XSD, FOAF, OWL
from owlrl import DeductiveClosure, RDFS_Semantics
import random, sys
import sys
import os
from string import Template
import re
import json

import requests
from bs4 import BeautifulSoup
from pprint import pprint

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os, random, json, logging, csv
import spotlight

import torch


"""
sequence of events into an article

schema.org information consistent with dbpedia or are we missing information like location date etc.

additional resources like wikidata.

truth assesment percentages or over subgraphs
"""

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

    truth_percentage_dbpedia = (consistency_state_dbpedia.count(True)/(consistency_state_dbpedia.count(True) + consistency_state_dbpedia.count(False))) * 100

    truth_percentage_wikidata = (consistency_state_wikidata.count(True)/(consistency_state_wikidata.count(True) + consistency_state_wikidata.count(False))) * 100

    return (
        truth_percentage_dbpedia,
        truth_percentage_wikidata
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

    mapping = None
    for s, p, o in wikidata_graph.triples(
            (
                URIRef(wikidata_ref_predicate[0]),
                URIRef("http://wikiba.se/ontology#directClaim"),
                None
            )
    ):
        if not mapping:
            mapping = o
            
    all_objects = []
    for s, p, o in wikidata_graph.triples(
            (
                wikidata_ref_subject[0],
                mapping,
                None
            )
    ):
        all_objects.append(
            o
        )

    consistency_found = False
    for object_ in all_objects:
        if wikidata_ref_object[0] in object_:
            consistency_found = True
            return True

    if not consistency_found:
        print("Inconsistency ", wikidata_ref_object[0], all_objects)
        return False
    else:
        return True
