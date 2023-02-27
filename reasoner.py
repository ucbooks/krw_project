from ast import arg
from rdflib import Graph, RDFS, RDF, URIRef, Namespace, Literal, XSD, FOAF
from owlrl import DeductiveClosure, RDFS_Semantics
import random, sys
import sys
import os
from string import Template
import re
import json


def reasoner(turtle_file):

    g = Graph()
    g.parse(turtle_file, format="turtle")

    consistency_state = []
    for subject, predicate, object_ in g.triples((None, None, None)):
        dbpedia_graph = Graph()

        uri = subject
        try:
            predicate = predicate
            object_ = object_

            print("URI succeeded")
            print(uri, predicate, object_)
            
            dbpedia_graph.parse(
                uri
            )

            for subj, pred, obj in dbpedia_graph.triples(
                (
                    uri,
                    predicate,
                    None
                )
            ):
                print("object printer")
                print(obj, object_)
                if str(obj) == str(object_):
                    consistency_state.append(True)
                else:
                    consistency_state.append(False)
        except:
            print("URI failed :", uri)

    return consistency_state
