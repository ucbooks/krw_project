from ast import arg
from rdflib import Graph, RDFS, RDF, URIRef, Namespace, Literal, XSD, FOAF, OWL
from owlrl import DeductiveClosure, RDFS_Semantics
import random, sys
import sys
import os
from string import Template
import re
import json


"""
sequence of events into an article

schema.org information consistent with dbpedia or are we missing information like location date etc.

additional resources like wikidata.

truth assesment percentages or over subgraphs
"""


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
    print("The object is")
    print(object_)
    try:
        dbpedia_graph_object.parse(
            object_
        )
    except:
        print("URI failed wikidata object parse: ", object_)
        return False

    wikidata_ref_object = []
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

    all_objects = []
    for s, p, o in wikidata_graph.triples(
            (
                None,
                None,
                None
            )
    ):
        if ("/prop/"+wikidata_ref_predicate[0].split("/")[-1] in p) and (s == wikidata_ref_subject[0]):
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
