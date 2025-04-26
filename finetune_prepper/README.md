### Finetune Prepper
Note: The tsql converter relies on auto-generated ANTLR files. If using this tool for the first time, run the command: `antlr4 -Dlanguage=Python3 ./tsql_converter/monadic_parser/newTSQL.g4` first before converting any files.

Example Usage: 
1. First, convert the raw json files from the [spider dataset](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view) to TSQL: `python3 ./tsql_converter/main.py --train_data=/home/ubuntu/data/spider_data/train_spider.json --dev_data=/home/ubuntu/data/spider_data/dev.json --test_data=/home/ubuntu/data/spider_data/test.json`
2. Then, add in the prompt and database information: `python3 prompt_adder.py --dev_file=/home/ubuntu/data/spider_data/dev_tsql.csv --test_file=/home/ubuntu/data/spider_data/test_tsql.csv --train_file=/home/ubuntu/data/spider_data/train_spider_tsql.csv --test_tables_file=/home/ubuntu/data/spider_data/test_tables.json --train_tables_file=/home/ubuntu/data/spider_data/tables.json`

Example Conversions:

Input: SQL  1
```sql
SELECT DISTINCT 
    T1.first_name, 
    T3.treatment_type_description
FROM 
    professionals AS T1
JOIN 
    Treatments AS T2 
    ON T1.professional_id = T2.professional_id
JOIN 
    Treatment_types AS T3 
    ON T2.treatment_type_code = T3.treatment_type_code;
```
Output: TSQL 1
```sql
PROJECT DISTINCT 
    T1.first_name, 
    T3.treatment_type_description
FROM 
    JOIN AS T1 FROM professionals
WITH 
    JOIN AS T2 FROM Treatments
WITH 
    AS T3 FROM Treatment_types
ON 
    T2.treatment_type_code = T3.treatment_type_code
ON 
    T1.professional_id = T2.professional_id;
```

Input: SQL 2
```sql
SELECT 
    AirportName 
FROM 
    Airports 
WHERE 
    AirportCode NOT IN (
        SELECT 
            SourceAirport 
        FROM 
            Flights 
        UNION 
        SELECT 
            DestAirport 
        FROM 
            Flights
    );
```
Output: TSQL 2
```sql
PROJECT 
    AirportName 
FROM 
    SELECT 
    WHERE 
        AirportCode NOT IN UNION
            PROJECT 
                SourceAirport 
            FROM 
                Flights 
            WITH 
            PROJECT 
                DestAirport 
            FROM 
                Flights
FROM 
    Airports;
```
Example Prompt: 
```
### System Prompt: You are an intelligent AI specialized in generating DSQL queries. DSQL is a domain-specific
language similar to SQL but with different ordering and syntax to make it closer to natural language. Your task
is to help users formulate DSQL queries to retrieve specific information from the tables below. Please convert
the following question into DSQL, keeping the table structure in mind. \n### Tables: CREATE TABLE club ( Club_ID
number, Name text, Manager text, Captain text, Manufacturer text, Sponsor text, PRIMARY KEY (Club_ID)); CREATE TABLE
player ( Player_ID number, Name text, Country text, Earnings number, Events_number number, Wins_count number,
Club_ID number, PRIMARY KEY (Player_ID), FOREIGN KEY (Club_ID) REFERENCES club (Club_ID)); \n### Question: Count the
number of clubs.\n### Response (DSQL):
```
