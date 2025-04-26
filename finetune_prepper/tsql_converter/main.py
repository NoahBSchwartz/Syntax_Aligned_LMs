"""
Code for converting SQL to TSQL
1. SQL to Intermediate:
    Main call goes to the conv_query(sql, stack_) which recursively creates a stack of the expressions in the right ordering
    Then call to create_tsql(stack_) creates the string representation of the contents in the stack
2. Intermediate to TSQL
    First call parser() which converts old TSQL into a parse tree
    Then clean_tree() reorders parse tree to be new TSQL ordering
    And stack_to_string() converts stack of old TSQL expressions into new TSQL string

Note: 2 phases were used because during development, we found it was far easier to make small changes 
to our current TSQL version than to continuously rewrite our initial translator. This results 
in a cleaner codebase (allowing for far faster fututure changes) where we reorder the intermediate 
representation, rather than combining translation and reordering into one step.
For all future changes, modify the `IntermediateToTsql` class
"""
from utils import SqlToIntermediate, IntermediateToTsql
import json
import pandas as pd
from sqlglot import parse_one
import os
import argparse
import regex as re
from tqdm import tqdm
import click

def translate_query(query: str) -> str:
    """Translate the given SQL query to its equivalent TSQL query"""
    SqlToInter = SqlToIntermediate()
    InterToTsql = IntermediateToTsql()

    stack_values = SqlToInter.conv_query(query, [])

    # conv_query deals with set operations and subqueries from start to end, automatically
    # running them through the final processing step (create_tsql). If the query does not
    # contain either of these, we need to run it through the final processing step.
    if "UNION" in stack_values or "INTERSECT" in stack_values or "EXCEPT" in stack_values:
        inter_tsql = stack_values
    elif "(SELECT" in str(parse_one(query)):
        inter_tsql = stack_values
    else:
        inter_tsql = SqlToInter.create_tsql(stack_values)

    search_tree = InterToTsql.parser(inter_tsql)
    tsql = InterToTsql.tree_to_query(search_tree)
    tsql = re.sub("  +", " ", tsql)

    return tsql

@click.command()
# Path to the original SPIDER data json files
@click.argument('files', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def main(
    files: list[str]
):
    """
    Accepts raw JSON files from the SPIDER dataset (ie. train.json, test.json...)
    containing the fields: 'query', 'db_id', and 'question'. It translates 'query'
    into a TSQL query, converting the JSON file into a CSV file with the columns
    'question', 'db', 'sql', and 'tsql'.
    """
    for file_path in tqdm(files, desc="Processing files"):
        # List of queries (33 total) in the SPIDER dataset that produce bugs either because they're valid
        # SQLite queries but invalid SQL queries or just because they're so long that we haven't had time
        # to fix them yet.
        ignore_list = [
            "T1.StuID  =  PRESIDENT_Vote",
            "T1.StuID  =  SECRETARY_Vote",
            "T1.StuID  =  Class_Senator_Vote",
            "T2.name  =  \'Boston Red Stockings\'",
            "t1.uk_vat_number",
            "T2.actid  =  T2.actid"
        ]

        vals = {}
        with open(file_path, 'r') as f:
            vals = json.load(f)
        output_vals = []
        for q in tqdm(vals, desc="Processing queries"):
            sql = q['query']
            db = q['db_id']
            question = q['question']
            # Remove enclosing quotes if present. Cannot just strip quotes since
            # they may be valid at the end of a query, such as:
            # `SELECT id FROM users WHERE name = "admin"`
            if sql[0] == "\"":
                sql = sql[1:-1]

            if not any(ignore_sql in sql for ignore_sql in ignore_list):
                tsql = translate_query(sql)
                output_vals.append({'question': question, 'db': db, 'sql': sql, 'tsql': tsql})

        df = pd.DataFrame(data=output_vals)
        df.to_csv(os.path.splitext(file_path)[0] + "_tsql.csv", index=False)


if __name__ == "__main__":
   main()