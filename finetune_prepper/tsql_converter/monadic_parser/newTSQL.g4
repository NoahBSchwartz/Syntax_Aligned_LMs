grammar newTSQL;

q
    : GROUPBY columnNameList FROM q #GROUPBY
    | HAVING comparison q #HAVING
    | PROJECT (DISTINCT)? columnNameList (fromClause)? #PROJECT
    | fromClause #FROMSELECTQ
    | orderByClause #ORDERBYQ
    | LIMIT numLiteral FROM (whereClause)? orderByClause #LIMIT
    | whereClause q #WHEREQ
    | (INTERSECT | EXCEPT | UNION) subQuery AND subQuery #SETOPERATIONS
    ;

joinClause: FROM JOIN subQuery (AS name)? AND subQuery (AS name)? ON columnName '=' columnName #JOIN;
orderByClause: ORDERBY ((columnName (ASC | DESC)?) (',' columnName (ASC | DESC)?)* | comparison) FROM q #ORDERBY;
whereClause: WHERE comparison #WHERE;
fromClause: FROM_SELECT subQuery (joinClause (joinClause)* | AS name | AS name joinClause (joinClause)*)? #FROMSELECT;

subQuery
    : name #NAME
    | q #SUBQUERY
    ;

comparison
    : variable ('>' | '>=' | '<' | '<=' | '=' | '<>') (variable | subQuery)
    | variable (NOT)? IN subQuery
    | variable (NOT)? LIKE (stringLiteral | subQuery)
    | variable BETWEEN (numLiteral | subQuery | stringLiteral | dateLiteral) AND (numLiteral | subQuery |  stringLiteral | dateLiteral)
    | count ('>' | '>=' | '<' | '<=' | '=' | '<>') (numLiteral | subQuery)
    | NOT comparison
    | comparison (AND | OR) comparison
    ;

variable
    : columnName
    | constant
    | constant ('+' | '-' | '*' | '/' | MOD) constant
    ;

count
    : COUNT '(*)'
    | COUNT '(' (DISTINCT)? columnName ')'
    ;

columnNameList
    : columnName (',' columnName)* 
    ;

columnName
    : name ('.' name)? 
    | columnName AS name
    | (MAX | MIN | AVG | SUM) '(' (DISTINCT)? columnName ')'
    | '(' columnName ')'
    | columnName ('+' | '-' | '*' | '/' | MOD) columnName
    | count
    | '*'
    ;

constant
    : numLiteral 
    | stringLiteral 
    | dateLiteral 
    | boolLiteral
    ;

LIMIT: 'LIMIT';
SELECT: 'SELECT';
COUNT: 'COUNT';
DISTINCT: 'DISTINCT';
FROM: 'FROM';
INTERSECT: 'INTERSECT OF';
FROM_SELECT: 'FROM SELECT';
AS: 'AS';
ASC: 'ASC';
DESC: 'DESC';
MAX: 'MAX';
EXCEPT: 'EXCEPT OF';
MIN: 'MIN';
AVG: 'AVG';
UNION: 'UNION OF';
WHERE: 'WHERE';
JOIN: 'JOIN';
SUM: 'SUM';
AND: 'AND';
ON: 'ON';
ORDERBY: 'ORDER BY';
GROUPBY: 'GROUP BY';
PROJECT: 'PROJECT';
NOT: 'NOT';
HAVING: 'HAVING';
IN: 'IN';
LIKE: 'LIKE';
MOD: 'MOD';
OR: 'OR';
BETWEEN: 'BETWEEN';

name: ID;
ID: [a-zA-Z][a-zA-Z0-9_]*;
numLiteral: INTEGER #NUMLITERAL | FLOAT #NUMLITERAL;
INTEGER: '-'? [0-9]+;
FLOAT: '-'? ([0-9]+ '.' [0-9]* | '.' [0-9]+);
stringLiteral: SINGLE_QUOTE_STRING | DOUBLE_QUOTE_STRING #STRINGLITERAL;
SINGLE_QUOTE_STRING: '\'' (~['\r\n] | '\'\'')* '\'';
DOUBLE_QUOTE_STRING: '"' (~["\r\n] | '""')* '"';
dateLiteral: DATE;
DATE: [0-9]{4} '-' [0-9]{2} '-' [0-9]{2};
boolLiteral: 'TRUE' | 'FALSE';
WS: [ \t\r\n]+ -> skip;