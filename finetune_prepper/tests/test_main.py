import sys
sys.path.insert(0, '../tsql_converter')
from main import translate_query

def test_translate_order_by_nested_within_project():
    """
    PROJECT clauses filter out columns that may be sorted by. We therefore require ORDER BY clauses to appear within
    PROJECT clauses so the ordering happens before the column filtering.
    """
    sql = "SELECT name FROM table ORDER BY age"
    tsql = "PROJECT name FROM ORDER BY age FROM table"
    assert translate_query(sql).strip() == tsql.strip()

def test_translate_order_by_and_project_subqueries():
    """
    PROJECT clauses filter out columns that may be sorted by. We therefore require ORDER BY clauses to appear within
    PROJECT clauses so the ordering happens before the column filtering.
    """
    sql = "SELECT name FROM (SELECT * FROM table ORDER BY age) ORDER BY height"
    tsql = "PROJECT name FROM ORDER BY height FROM PROJECT * FROM ORDER BY age FROM table"
    assert translate_query(sql).strip() == tsql.strip()
