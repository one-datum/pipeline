#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import astropy.table as at

parser = argparse.ArgumentParser()
parser.add_argument("--reference", required=True, type=str)
parser.add_argument("--table", required=True, type=str)
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--figure", required=True, type=str)
args = parser.parse_args()


reference = at.Table.read(args.reference)
table = at.Table.read(args.table)

joined = at.join(reference, table, keys=["source_id"])
print(len(reference), len(joined), len(table))
print(reference["source_id"])
assert 0
