import enum
import json
import re
import csv
from tqdm import tqdm
import os
import click

class DatasetSplit(enum.StrEnum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"

@click.command()
@click.option('--file',
              required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Path to a TSQL csv file')
@click.option('--tables_file',
              required=True,
              type=click.Path(exists=True, dir_okay=False),
              help='Path to the Spider SQL tables file containing the schemas for this split. Typically either `tables.json` or `test_tables.json`.')
@click.option('--split',
              required=True,
              type=click.Choice(DatasetSplit, case_sensitive=False),
              help="Which dataset split the given file is for. Useful for uniquely identifying test cases across splits.")
def add_prompts(
    file: str,
    tables_file: str,
    split: DatasetSplit,
):
    if not tables_file.endswith(".json"):
        print(f"Error: {tables_file} does not have .json extension")
        return
    if not file.endswith('.csv'):
        print(f"Error: {file} is not a CSV file")
        return
    
    with open(tables_file, "r") as table_file:
        tables_data = json.load(table_file)
        database_schemas = {entry["db_id"]: entry for entry in tables_data}

    output_vals = process_file(file, database_schemas, split)
    output_file = os.path.splitext(file)[0] + "_cleaned.json"
    with open(output_file, 'w') as out_file:
        json.dump(output_vals, out_file, indent=2)
    print(f"Processed data from {file} has been saved to {output_file}")

def standardize_database_schema(schema):
    """
    Converts a SQLite schema dump (in the form of DDL statements) into a simplified
    DDL that just includes table and column names, data types, primary and foreign
    key specifications, and column constraints. This simplified DDL contains all the
    information required for the LLM to infer identifiers and column relationships,
    without bogging it down with too many extraneous details.
    """
    schema = schema.replace("`", "").replace('"', '')
    schema = re.sub(r'\s+DEFAULT\s+(?:NULL|\'\')', '', schema)
    schema = re.sub(r'PRIMARY KEY \((.*?)\)', r'PRIMARY KEY(\1)', schema)
    schema = re.sub(r'FOREIGN KEY \((.*?)\)', r'FOREIGN KEY(\1)', schema)
    schema = re.sub(r',\s*CONSTRAINT\s+\w+\s+CHECK\s*\(.*?\)', '', schema) 
    schema = re.sub(r'-- phpMyAdmin SQL Dump -- version 4.0.10.7 -- http:\/\/www.phpmyadmin.net -- -- Host: localhost -- Generation Time: Mar 20, 2015 at 01:43 AM -- Server version: 5.5.34-cll-lve -- PHP Version: 5.4.23    \/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT *\/; \/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS *\/; \/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION *\/; \/*!40101 SET NAMES utf8 *\/;', '', schema)    
    schema = schema.replace(' NOT NULL', '')
    schema = re.sub(r'VARCHAR\(\d+\)', 'VARCHAR', schema)
    schema = re.sub(r'INTEGER\s+NOT\s+NULL', 'INTEGER', schema)
    schema = re.sub(r'CREATE\s+(UNIQUE\s+)?INDEX.*?;', '', schema)
    schema = re.sub(r'\s+', ' ', schema)
    return schema.strip()

def create_table_sql(table_index, data, table_name_original):
    column_names = data["column_names"]
    column_names_original = data["column_names_original"]
    column_types = data["column_types"]
    foreign_keys = data["foreign_keys"]
    primary_keys = data["primary_keys"]
    table_names_original = data["table_names_original"]
    columns = []
    primary_keys_list = []
    foreign_keys_list = []
    for idx, (col_index, col_name) in enumerate(column_names):
        if col_index == table_index:
            col_name_original = column_names_original[idx][1]
            col_type = column_types[idx]
            col_def = f'"{col_name_original}" {col_type}'
            if idx in primary_keys:
                primary_keys_list.append(f'"{col_name_original}"')
            columns.append(col_def)
    for fk in foreign_keys:
        if (
            fk[0] in range(len(column_names))
            and column_names[fk[0]][0] == table_index
        ):
            referenced_table = table_names_original[column_names[fk[1]][0]]
            referenced_column = column_names_original[fk[1]][1]
            foreign_column = column_names_original[fk[0]][1]
            foreign_keys_list.append(
                f'FOREIGN KEY ("{foreign_column}") REFERENCES "{referenced_table}" ("{referenced_column}")'
            )
    table_statement = (
        f'CREATE TABLE "{table_name_original}" (\n    '
        + ",\n    ".join(columns)
    )
    if primary_keys_list:
        table_statement += (
            f',\n    PRIMARY KEY ({", ".join(primary_keys_list)})'
        )
    if foreign_keys_list:
        table_statement += ",\n    " + ",\n    ".join(foreign_keys_list)
    table_statement += "\n);\n\n"
    return table_statement
    
def generate_sql(data):
    """
    Given a database schema, this function iterates through all tables 
    in the db and calls create_table_sql to convert each table into a 
    simplified DDL

    `data` contains the database schema info associated with a particular db_id in the format:
        column_names
        column_names_original
        column_types
        foreign_keys
        primary_keys
        table_names_original
    This represents a single row of the database_schemas variable
    """
    table_names = data["table_names"]
    table_names_original = data["table_names_original"]
    sql_output = ""
    for i, table_name in enumerate(table_names):
        sql_output += create_table_sql(i, data, table_names_original[i])
    return sql_output

def process_file(input_file, database_schemas, split):
    """
    `input_file` should be a path to a CSV file with at least 4 entries in each row in the following order:
    1) Natural language question that a query should be synthesized to answer
    2) Database name to use for this question
    3) Expected SQL query
    4) Equivalent TSQL query
    This file can be generated by running the original SPIDER dataset json files through the tsql_converter

    `database_schemas` should contain the data of the tables.json file from the original SPIDER dataset with entries:
        column_names
        column_names_original
        column_types
        foreign_keys
        primary_keys
        table_names_original
    Note: in this code, database_schemas is in the form of a dictionary keyed by db_id

    split should be one of `train`, `test`, or `validation`. The split and test case index should uniquely identify each test case.
    """
    output_vals = []
    with open(input_file, "r") as file:
        reader = csv.reader(file)
        rows = list(reader)[1:]
    
    for idx, row in tqdm(enumerate(rows), desc=f"Processing {input_file}"):
        question = row[0]
        db_id = row[1]
        try:
            result_sql = generate_sql(database_schemas[db_id])
        except KeyError:
            print(f"Database {db_id} not found, skipping")
            continue
        
        if result_sql:
            # Standardize the table data to make it more readable
            clean = (
                result_sql.replace("\n", " ")
                .replace("\t", " ")
                .replace('"', " ")
                .replace(",", ", ")
                .replace("( ", "(")
                .replace(" )", ")")
            )
            tables = " ".join(clean.split())
            standardized_schema = standardize_database_schema(tables)
            prompt = f"""### System Prompt: You are an intelligent AI specialized in generating TSQL queries. TSQL is a domain-specific language similar to SQL but with different ordering and syntax to make it closer to natural language. The ordering format or precedence of the SQL language tokens are as follows: LIMIT -> UNION/INTERSECT/EXCEPT -> PROJECT/PROJECT DISTINCT -> ORDER BY -> WHERE (HAVING) -> AGGREGATE/GROUP BY -> WHERE (WHERE) -> JOIN -> AS -> table -> subquery. This means that if there is a LIMIT clause in the SQL query, the entire clause will be placed at the beginning in the TSQL output. If there is an PROJECT clause, it will be placed after the LIMIT or UNION clause (assuming either one of them appear in the SQL query, if not PROJECT clause will be at the beginning), and so on. Your task is to help users formulate TSQL queries to retrieve specific information from the tables below. Please convert the following question into TSQL, keeping the table structure in mind. \n### Tables: {standardized_schema} \n### Question: {question}\n### Response (TSQL):"""
            output_vals.append({"prompt": prompt, "sql": row[2], "dsql": row[3], "database": row[1], "index": idx, "split": split})
    
    return output_vals

if __name__ == "__main__":
    add_prompts()