from ast import arg
from rdflib import Graph, RDFS, RDF, URIRef, Namespace, Literal, XSD, FOAF
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
        Check the validity of the turtle triples
           - input: Turtle file
           - output: Consistency booleans for each triple
    """

    g = Graph()
    g.parse(turtle_file, format="turtle")

    consistency_state = []
    for subject, predicate, object_ in g.triples((None, None, None)):
        dbpedia_graph = Graph()

        uri = subject
        try:
            predicate = predicate
            object_ = object_
            
            dbpedia_graph.parse(
                uri
            )

            all_objects = []
            for object__ in dbpedia_graph.objects(
                    subject=uri,
                    predicate=predicate,
                    unique=True
            ):
                all_objects.append(
                    object__
                )

            if object_ in all_objects:
                consistency_state.append(
                    True
                )
            else:
                consistency_state.append(
                    False
                ) 
        except:
            print("URI failed :", uri)
            consistency_state.append(False)

    return consistency_state
