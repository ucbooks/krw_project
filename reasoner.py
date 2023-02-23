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

    
    
