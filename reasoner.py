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
from rdflib import Graph, URIRef, RDF
uri = URIRef('http://dbpedia.org/resource/Richard_Nixon')
person = URIRef('http://dbpedia.org/ontology/Person')

g = Graph()
g.parse(uri)

for obj in g.objects(subject=uri, predicate=RDF.type):
    if obj == person:
        print uri, "is a", person
"""


def reasoner(turtle_file):

    g = Graph()
    g.parse(turtle_file, format="turtle")

    consistency_state = []
    for subject, predicate, object_ in g.triples((None, None, None)):
        dbpedia_graph = Graph()

        uri = subject
        predicate = predicate
        object_ = object_

        print(uri)
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
            if str(obj) == str(object_):
                consistency_state.append(True)
            else:
                consistency_state.append(False)

    return consistency_state
