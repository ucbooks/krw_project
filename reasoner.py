from ast import arg
from rdflib import Graph, RDFS, RDF, URIRef, Namespace, Literal, XSD
from owlrl import DeductiveClosure, RDFS_Semantics
import random, sys
import sys
import os
import pandas as pd
from string import Template
import re
import json


def reasoner(turtle_file):

    """
        Takes in a graph and performs a search on dbpedia for the unknown classes. 
    """

    with open(turtle_file, "r") as hander:
        turtle_data = handler.read()

    # Split the statements by fullstop.
    statements = turtle_data.split(".")

    # Strip statements
    stripped_statements = []
    for statement in statements:
        stripped_statements.append(
            statement.strip()
        )

    # Turn statements into (subject, predicate, object) tuples
    spo_tuples = []
    for statement in stripped_statements:
        split_statement = statement.split(" ")
        spo_tuples.append(
            (
                split_statement[0].strip(),
                split_statement[1].strip(),
                split_statement[2].strip()
            )
        )

    dbpedia_graph = Graph()
    dbpedia_graph.parse("https://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&qtxt=select%20distinct%20%3FConcept%20where%20%7B%5B%5D%20a%20%3FConcept%7D%20LIMIT%20100&format=text%2Fhtml&timeout=30000&signal_void=on&signal_unconnected=on")

    # Query DBPedia to check the consistency of each tuple.
    consistency_state = []
    for tuple_ in spo_tuples:
        subject = tuple_[0]
        predicate = tuple_[1]
        object_ = tuple_[2]

        
        
    
