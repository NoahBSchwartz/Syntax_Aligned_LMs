from antlr4 import *
from monadic_parser.newTSQLLexer import newTSQLLexer
from monadic_parser.newTSQLParser import newTSQLParser
import re
import sqlglot


class SqlToIntermediate:
    """
    SQL to TSQL Converter (First Generation)

    This class contains the first sql_to_tsql converter we created,
    converting standard SQL to an older version of TSQL.

    Key Methods:
        - conv_query(): Parse and stack SQL expressions
        - create_tsql(): Convert stacked expressions to TSQL

    Note:
        We convert into an older version of TSQL with this class and
        use another class to convert to the newest version of TSQL.
        During development, we found it was far easier to make 
        small changes to our current TSQL version than to continuously
        rewrite our initial translator.
    """

    @classmethod
    def check_insert(self, stack, obj, obj_type):
        """
        Reorders stack maintaining operator ordering when inserting new object.
        
        Args:
            stack (list): Current expression stack
            obj: New object to insert 
            obj_type: Type of object being inserted
        Returns: Reordered stack with new object in correct position
        """
        n_stack = []
        for items in reversed(stack):
            if (
                obj_type == sqlglot.expressions.Where
                or obj_type == sqlglot.expressions.Having
            ) and (
                type(items) != sqlglot.expressions.Limit
                and type(items) != sqlglot.expressions.Union
            ):
                n_stack.append(items)
                stack.pop()
            elif (
                obj_type == sqlglot.expressions.Order
                or obj_type == sqlglot.expressions.Group
            ) and (
                type(items) != sqlglot.expressions.Limit
                and type(items) != sqlglot.expressions.Union
                and type(items) != sqlglot.expressions.Where
            ):
                n_stack.append(items)
                stack.pop()
            elif obj_type == sqlglot.expressions.Union and (
                type(items) != sqlglot.expressions.Limit
            ):
                n_stack.append(items)
                stack.pop()
            else:
                continue
        stack.append(obj)
        for items in reversed(n_stack):
            stack.append(items)
        del n_stack
        return stack

    @classmethod
    def parse_set_operations(self, node):
        if (
            "UNION" not in str(node)
            and "INTERSECT" not in str(node)
            and "EXCEPT" not in str(node)
        ):  # Recursion base case: if the query contains a subquery, send it
            # back through the whole process. Otherwise, just rearrange it and 
            # return the final TSQL string.
            if "(SELECT" in str(node):
                return self.conv_query(str(node), [])
            else:
                return self.create_tsql(self.conv_query(str(node), []))

        words = re.split(r"(\s|\(|\))", str(node))
        for word in words:
            if word == "UNION":
                item1, item2 = str(node).split("UNION")
                q = "UNION OF "
                break
            if word == "INTERSECT":
                item1, item2 = str(node).split("INTERSECT")
                q = "INTERSECT OF "
                break
            if word == "EXCEPT":
                item1, item2 = str(node).split("EXCEPT")
                q = "EXCEPT OF "
                break

        for e in [item1]:
            q += str(self.parse_set_operations(e)) + " AND "

        for e in [item2]:
            q += str(self.parse_set_operations(e)) + " AND "

        q = re.sub(r" {2,}", " ", q)
        q = q.strip()
        # We added an "AND " to the end of q when we were building
        # up the query. Remove it for the final return statement.
        return q[:-4]

    @classmethod
    def parse_subquery(self, sql):
        """
        This method handles both subqueries and set operations (which always contain 
        subqueries) since they require similar recursive parsing. If neither is present,
        the query is sent through normal processing via create_tsql.
        """
        if "(SELECT" not in str(sql):
            if "UNION" in str(sql) or "INTERSECT" in str(sql) or "EXCEPT" in str(sql):
                return self.conv_query(sql, [])
            else:
                return self.create_tsql(self.conv_query(sql, []))

        words = re.split(r"(\s|\(|\))", str(sql))
        pieces = []
        inside_paren = False
        open_p_count = 0
        closed_p_count = 0
        piece_counter = 0
        prev_word = ""
        for word in words:
            pieces.append(word)
            if prev_word == "(" and word == "SELECT" and not inside_paren:
                # For current piece up to "(SELECT" keyword, join all tokens into single string
                pieces[piece_counter] = "".join(pieces[piece_counter:-2])
                # Add back (SELECT as separate token since we need to track subquery boundaries
                pieces.insert(piece_counter + 1, "(SELECT ")
                piece_counter += 2
                # Trim excess tokens that were already joined into earlier pieces
                pieces = pieces[: -(len(pieces) - piece_counter)]
                inside_paren = True
            if word == "(":
                open_p_count += 1
            elif word == ")":
                closed_p_count += 1
                if closed_p_count == open_p_count and inside_paren:
                    pieces[piece_counter - 1] = "".join(pieces[(piece_counter - 1) :])
                    pieces = pieces[: -(len(pieces) - piece_counter)]
                    inside_paren = False
            prev_word = word
        pieces[piece_counter] = "".join(pieces[(piece_counter):])
        processed_pieces = pieces[: piece_counter + 1]

        # prcoessed_pieces alternates between regular SQL fragments and subquery strings
        # e.g. ["SELECT * FROM", "(SELECT id FROM users)", "WHERE count >", "(SELECT max(count) FROM stats)"]
        base_query = []  
        subqueries = []
        for i in range(0, len(pieces), 2):  # Step by 2 to process pairs of regular SQL and subqueries
            if i < len(processed_pieces):
                base_query.append(processed_pieces[i])
                if i + 2 < len(processed_pieces):
                    base_query.append("Subquery_Standin")
                    # [1:-1] removes parentheses before parsing the subquery
                    subqueries.append(self.parse_subquery(processed_pieces[i + 1][1:-1]))
        base_query = str(self.create_tsql(self.conv_query("".join(base_query), [])))
        items = base_query.split("Subquery_Standin")
        query = items[0]
        for i in range(1, len(items)):
            query += str(subqueries[i - 1]) + items[i]
        return " ".join(query.split())

    @classmethod
    def handle_joins(self, node, q, join_array, main_table):
        """
        Processes JOIN nodes and builds JOIN clauses in TSQL format.
        
        Args:
            node: Current JOIN node
            q (str): Query string being built
            join_array (list): Array of processed JOIN tables
            main_table (str): Primary table being queried
        Returns: Query string with formatted JOIN clause
        """
        last_part = ""
        q_node = node.on()
        if join_array == []:
            q += " FROM JOIN "
        if main_table not in join_array:
            join_array.append(main_table)
        if node.this not in join_array:
            join_array.append(str(node.this))
        if "=" in str(q_node):
            if not str(q_node).startswith("ON"):
                pattern = r"\bJOIN\b.*?\bON\b"
                last_part = re.sub(pattern, "", str(q_node)).strip()
            words = last_part.split()
            append_join = False
            i = 0
            for word in words:
                if "OR" in words:
                    if "." in word:
                        key = word.split(".")[0]
                        for entries in join_array:
                            entry = str(entries).split()
                            if key.lower() == entry[-1].lower():
                                q += entries
                                break
                        if not append_join:
                            q += " AND "
                            append_join = True
                        else:
                            q += " ON " + " ".join(words) + "__________"
                            break
                else:
                    if "." in word:
                        key = word.split(".")[0]
                        for entries in join_array:
                            entry = str(entries).split()
                            if key.lower() == entry[-1].lower():
                                q += entries
                                break
                        if not append_join:
                            q += " AND "
                            append_join = True
                        else:
                            try:
                                q += (
                                    " ON "
                                    + words[i - 2]
                                    + " "
                                    + words[i - 1]
                                    + " "
                                    + words[i]
                                    + " FROM JOIN "
                                )
                            except:
                                q += (
                                    " ON "
                                    + words[i - 2]
                                    + " "
                                    + words[i - 1]
                                    + " FROM JOIN "
                                )
                            append_join = False
                i += 1
        return q

    @classmethod
    def conv_query(self, sql, stack):
        """
        Converts normal sql query into a stack of expressions which is ordered 
        according to the inter_tsql rules.
        
        First, break the query into parts (using sqlglot). For each part, call a specific 
        function to correctly reorder and reconfigure the stack.
        """

        parsed_sql = sqlglot.parse_one(sql)

        for node in parsed_sql.walk(bfs=True):
            n = type(node)
            match n:
                case sqlglot.expressions.Subquery:
                    if "(SELECT" in str(node):
                        q = self.parse_subquery(parsed_sql)
                        return q
                case (
                    sqlglot.expressions.Union
                    | sqlglot.expressions.Intersect
                    | sqlglot.expressions.Except
                ):
                    q = self.parse_set_operations(node)
                    return q
                case sqlglot.expressions.Limit:
                    if len(stack) == 0:
                        stack.append(node)
                    else:
                        stack.insert(0, node)
                case sqlglot.expressions.Order:
                    stack = self.check_insert(stack, node, type(node))
                case sqlglot.expressions.Group:
                    stack = self.check_insert(stack, node, type(node))
                case sqlglot.expressions.Distinct:
                    stack = self.check_insert(stack, node, type(node))
                case sqlglot.expressions.Select:
                    stack.append(node)
                case sqlglot.expressions.From:
                    if type(node.this) == sqlglot.expressions.Subquery:
                        continue
                    else:
                        stack.append(node)
                case sqlglot.expressions.Where | sqlglot.expressions.Having:
                    if len(stack) == 0:
                        stack.append(node)
                    else:
                        stack = self.check_insert(stack, node, type(node))
                case sqlglot.expressions.Join:
                    stack.insert(len(stack), node)
                case _:
                    continue

        return stack

    @classmethod
    def create_tsql(self, stack):
        """
        Converts a stack which is ordered according to the inter_tsql rules into an actual
        inter_tsql query.
        
        For each node in the stack, call a specific function to convert the sql syntax into 
        inter_tsql syntax.
        """
        q = ""
        main_table = ""
        join_array = []

        for node in stack:
            match type(node):
                case sqlglot.expressions.Limit:
                    q += str(node) + " FROM "
                case (
                    sqlglot.expressions.Union
                    | sqlglot.expressions.Intersect
                    | sqlglot.expressions.Except
                ):
                    q += str(node) + " FROM "
                case sqlglot.expressions.Where | sqlglot.expressions.Having:
                    if type(node) == sqlglot.expressions.Where:
                        insert = " WHERE "
                    else:
                        insert = " HAVING "

                    substrings = str(node.this).split(" ")
                    reordered_not = []
                    i = 0
                    while i < len(substrings):
                        if substrings[i] == "NOT":
                            reordered_not.append(substrings[i + 1] + " NOT ")
                            i = i + 2
                        else:
                            reordered_not.append(substrings[i])
                            i = i + 1
                    reordered_not =  " ".join(reordered_not)

                    if "SELECT" in str(node.this):
                        main_table = ""
                        q += insert + str(reordered_not)
                    else:
                        q += insert + str(reordered_not) + " "
                case sqlglot.expressions.Order | sqlglot.expressions.Group:
                    q += str(node) + " FROM "
                case sqlglot.expressions.Distinct:
                    if str(node) == "DISTINCT":
                        loc_ = max(
                            [m.start() for m in re.finditer("PROJECT", q)]
                        ) + len("PROJECT")
                        q = q[:loc_] + " DISTINCT " + q[loc_:]
                    # q += " DISTINCT "
                case sqlglot.expressions.From:
                    main_table = str(node.this)
                    q += " SELECT " + str(node.this)
                case sqlglot.expressions.Select:
                    q += " PROJECT "
                    for expr in node.expressions:
                        q += str(expr).replace("'", "") + ", "
                    q = q[:-2]
                    q += " FROM "
                case sqlglot.expressions.Join:
                    q = self.handle_joins(node, q, join_array, main_table)

        q = re.sub(r" {2,}", " ", q)
        # After the tsql translation takes place, only the string "FROM JOIN" may be left.
        # We'll remove it to complete the translation.
        if "FROM JOIN" in q[-10:]:
            q = q[:-10]
        q = q.strip()
        return q


class IntermediateToTsql:
    """
    Old TSQL to New TSQL Converter

    This class contains the second converter we created,
    converting an older version of TSQL to the newer version.
    Note: the code is structured this way because during development,
          we found it was easier to make small changes to our current 
          TSQL version than to rewrite our initial translator.

    Key Methods:
    - parser(): Converts old TSQL into a parse tree
    - clean_tree(): Reorders parse tree to be new TSQL ordering
    - stack_to_string(): Converts stack of old TSQL expressions into new TSQL string
    """

    def get_source_text_for_context(self, node):
        """
        Extracts the original source text from a parse tree node.
        
        Args: node: ANTLR parse tree node (either Terminal or NonTerminal)
        Returns: Original source text corresponding to this node from the input stream
        """
        if isinstance(node, TerminalNode):
            start_token = node.symbol
            stop_token = node.symbol
        else:
            start_token = node.start
            stop_token = node.stop if node.stop else node.start

        input_stream = start_token.getInputStream()
        start_index = start_token.start
        stop_index = stop_token.stop

        return input_stream.getText(start_index, stop_index)

    def build_tree(self, node, depth=0, parent_index=-1):
        tree = []

        def add_node(node, depth, parent_index):
            if node is None:
                return
            current_name = self.get_source_text_for_context(node)
            node_type = str(type(node).__name__)
            # Skip terminal nodes and empty nodes (empty nodes are valid in ANTLR
            # when a rule completes without consuming tokens)
            if node_type != "TerminalNodeImpl" and current_name:
                node_info = {
                    "type": node_type,
                    "name": current_name,
                    "depth": depth,
                    "parent_index": parent_index,
                }
                current_index = len(tree)
                tree.append(node_info)
                if hasattr(node, "children") and node.children:
                    for child in node.children:
                        add_node(child, depth + 1, current_index)

        add_node(node, depth, parent_index)
        return tree

    def clean_tree(self, search_tree):
        """
        Filters parse tree to keep only relevant TSQL operation nodes.
        
        Args: search_tree (list): List of node dictionaries from parser 
        Returns:Filtered tree containing only TSQL operation nodes
        """
        i = 0
        kept_nodes = [
            "PROJECT",
            "WHERE",
            "GROUPBY",
            "ORDERBY",
            "ErrorN",
            "SUBQUERY",
            "HAVING",
            "JOIN",
            "LIMIT",
            "SETOPERATIONS",
            "FROMSELECT",
        ]
        while i < len(search_tree):
            # Node types from ANTLR parser end in "Context" - remove this suffix
            if search_tree[i]["type"][:-7] not in kept_nodes:
                search_tree.pop(i)
            else:
                i += 1
        return search_tree

    def clean_nodes(self, search_tree):
        """
        Removes redundant nested expressions from node names in parse tree.
        
        Args: search_tree (list): List of node dictionaries
        Returns:Tree with simplified node names
        """
        i = 0
        max_depth = float("inf")
        while i < len(search_tree):
            starting_depth = search_tree[i]["depth"]
            x = i + 1
            while x < len(search_tree) and search_tree[x]["depth"] > starting_depth:
                if search_tree[x]["depth"] <= max_depth:
                    substring = search_tree[x]["name"]
                    main_string = search_tree[i]["name"]
                    if substring in main_string:
                        # Node types from ANTLR parser end in "Context" - remove this suffix
                        start_index = main_string.index(substring)
                        if search_tree[x]["type"][:-7] == "SUBQUERY":
                            search_tree[i]["name"] = (
                                main_string[:start_index]
                                + "SUBQUERY "
                                + main_string[start_index + len(substring) :].lstrip()
                            )
                            max_depth = search_tree[x]["depth"]
                        else:
                            search_tree[i]["name"] = (
                                main_string[:start_index]
                                + main_string[start_index + len(substring) :].lstrip()
                            )
                x += 1
            i += 1
            max_depth = float("inf")
        return search_tree

    def build_stack(self, search_tree, index=0, depth=0):
        """
        Converts parse tree into stack of TSQL expressions with subqueries.

        Args:
            search_tree (list): List of node dictionaries 
            index (int): Current position in tree
            depth (int): Current depth in tree
        Returns: (List of [type, expression, subqueries], next index)
        """
        stack = []
        while index < len(search_tree):
            current = search_tree[index]
            if current["depth"] < depth:
                return stack, index
            if current["type"][:-7] == "SUBQUERY":
                substack, new_index = self.build_stack(
                    search_tree, index + 1, current["depth"] + 1
                )
                if stack and isinstance(stack[-1][2], list):
                    stack[-1][2].append(substack)
                elif stack and stack[-1][2] is None:
                    stack[-1][2] = [substack]
                else:
                    stack.append([current["type"], current["name"], [substack]])
                index = new_index
            else:
                stack.append([current["type"], current["name"], None])
                index += 1
        return stack, index

    # LIMIT -> UNION/INTERSECT/EXCEPT -> PROJECT/PROJECT DISTINCT -> ORDER BY -> WHERE (HAVING) -> AGGREGATE/GROUP BY -> WHERE (WHERE) -> JOIN -> AS -> table -> subquery
    def reorder_stack(self, stack):
        type_order = [
            "LIMIT",
            "SETOPERATIONS",
            "PROJECT",
            "DISTINCT",
            "ORDERBY",
            "HAVING",
            "AGGREGATE",
            "GROUPBY",
            "WHERE",
            "JOIN",
            "FROMSELECT",
        ]
        type_to_order = {}
        for index, item_type in enumerate(type_order):
            type_to_order[item_type] = index
        reordered_stack = []
        for current_type in type_order:
            for item in stack:
                if item[0][:-7] == current_type:
                    reordered_stack.append(item)
        for item in stack:
            if item[0][:-7] not in type_order:
                reordered_stack.append(item)
        return reordered_stack

    def find_children(self, tree, node_type):
        """
        Gets immediate child nodes of specified type from parse tree.
        
        Args:
            tree (list): List of node dictionaries
            node_type (str): Type of nodes to find
        Returns: list: Child nodes matching requested type
        """
        parent_indices = []
        for i, node in enumerate(tree):
            if node["type"] == node_type:
                parent_indices.append(i)

        children = []
        for parent_idx in parent_indices:
            parent_depth = tree[parent_idx]["depth"]
            for node in tree:
                if (
                    node["parent_index"] == parent_idx
                    and node["depth"] == parent_depth + 1
                ):
                    children.append(node)
        return children

    def aggregate_children(self, tsql_string, children, aggs_in_query):
        """
        Processes aggregate functions in child nodes and updates query.
        
        Args:
            tsql_string (str): Current query string
            children (list): List of child nodes
            aggs_in_query (list): Tracked aggregate functions  
        Returns: (Updated query string, Updated aggregates list)
        """
        aggs = ["COUNT", "DISTINCT", "MAX", "MIN", "AVG", "SUM"]
        for child in children:
            for agg in aggs:
                if agg in child["name"]:
                    aggs_in_query.append(child["name"])
            child_lower = child["name"].lower()
            split_result = child_lower.split(" as ")
            if len(split_result) > 1:
                aggs_in_query.append(child["name"])
                original_after_as = child_lower[len(split_result[0]) + 4 :]
                tsql_string = tsql_string.replace(child["name"], original_after_as, 1)
        return tsql_string, aggs_in_query

    def merge_similar_triplets(self, triplets):
        """
        Combines JOIN conditions between same table pairs to simplify query.
        Merging is the first step to converting a JOIN condition to TSQL.
        
        Args: List of [table1, table2, condition] items
        Returns: Merged triplets with combined conditions
        """
        merged = {}
        for triplet in triplets:
            pair = frozenset([triplet[0], triplet[1]])
            identifier = triplet[2]
            if pair in merged:
                existing_triplet = merged[pair]
                existing_triplet[2] = f"{existing_triplet[2]} and {identifier}"
            else:
                merged[pair] = [triplet[0], triplet[1], identifier]
        result = list(merged.values())
        return result

    def chain_triplets(self, triplets):
        """
        Orders JOIN triplets to form a connected chain of table relationships
        
        Args: List of [table1, table2, condition] items   
        Returns: Ordered triplets forming optimal JOIN chain
        """
        if not triplets:
            return []
        connections = {}
        for i, (a, b, t) in enumerate(triplets):
            if a not in connections:
                connections[a] = []
            if b not in connections:
                connections[b] = []
            connections[a].append((b, t, i))
            connections[b].append((a, t, i))
        start = min(connections.keys(), key=lambda x: len(connections[x]))
        used_edges = set()
        result = []
        current = start
        while len(result) < len(triplets):
            next_connection = None
            for neighbor, t, idx in connections[current]:
                edge = tuple(sorted([current, neighbor]))
                if edge not in used_edges:
                    next_connection = (neighbor, t, edge)
                    break

            if next_connection is None:
                for node in connections:
                    for neighbor, t, idx in connections[node]:
                        edge = tuple(sorted([node, neighbor]))
                        if edge not in used_edges:
                            current = node
                            next_connection = (neighbor, t, edge)
                            break
                    if next_connection:
                        break

            neighbor, t, edge = next_connection
            used_edges.add(edge)
            result.append([current, neighbor, t])
            current = neighbor

        return result

    def triplets_to_string(self, triplets):
        """
        Formats JOIN triplets into TSQL string with proper JOIN and ON clauses.
        
        Args: List of [table1, table2, condition] JOIN specifications
        Returns: Formatted JOIN string with correct syntax
        """
        result_parts = []
        onclauses = []
        for i, (first, second, onclause) in enumerate(triplets):
            if i == 0:
                result_parts.append(first)
            if i < len(triplets) - 1:
                result_parts.append(" WITH JOIN ")
            else:
                result_parts.append(" WITH ")
            result_parts.append(second)
            onclauses.append(onclause)
        for onclause in reversed(onclauses):
            result_parts.append(" ON ")
            result_parts.append(onclause)
        return " ".join(result_parts)

    def stack_to_string(self, tsql_stack):
        """
        Converts stack of [type, expression, subqueries] triplets to TSQL string
        
        Args: List of processed TSQL expressions  
        Returns: Complete TSQL query with correct syntax
        """
        aggs_in_query = []
        tsql_stack = self.reorder_stack(tsql_stack)
        for i, item in enumerate(tsql_stack):
            tsql_string = item[1]
            match item[0][:-7]:
                case "WHERE": 
                    # WHERE clauses are simple since they operate on raw columns
                    tsql_string = tsql_string.replace("WHERE", "SELECT WHERE")
                case "HAVING":
                    # HAVING operates on aggregated values, so we need to extract
                    # the aggregate functions and add them to our tracking list
                    # We find all comparison expressions (e.g. "COUNT(*) > 5")
                    subtree = self.parser(tsql_string)
                    children = self.find_children(subtree, "ComparisonContext")
                    tsql_string, aggs_in_query = self.aggregate_children(
                        tsql_string, children, aggs_in_query
                    )
                    tsql_string = tsql_string.replace("HAVING", "SELECT WHERE")                
                case "JOIN":
                    tsql_string = tsql_string.replace("AND", "WITH")
                    words = tsql_string.split()
                    for i, word in enumerate(words):
                        if word == "AS":
                            table_alias = words[i + 1]
                            table_name = words[i - 1]
                            words[i - 1] = "AS " + table_alias
                            words[i] = "FROM"
                            words[i + 1] = table_name
                    tsql_string = " ".join(words)
                case "SETOPERATIONS":
                    tsql_string = tsql_string.replace("AND", "WITH")
                    tsql_string = tsql_string.replace("OF", "")
                case "FROMSELECT":
                    if tsql_stack[i - 1][0][:-7] == "JOIN":
                        tsql_stack.remove(item)
                    else:
                        tsql_string = tsql_string.replace(" SELECT", "")
                    words = tsql_string.split()
                    for i, word in enumerate(words):
                        if word == "AS":
                            table_alias = words[i + 1]
                            table_name = words[i - 1]
                            words[i - 1] = "AS " + table_alias
                            words[i] = "FROM"
                            words[i + 1] = table_name
                    tsql_string = " ".join(words)
                case "ORDERBY" | "LIMIT":
                    subtree = self.parser(tsql_string)
                    children = self.find_children(subtree, "ORDERBYContext")
                    tsql_string, aggs_in_query = self.aggregate_children(
                        tsql_string, children, aggs_in_query
                    )
                    words = tsql_string.split()
                    word = ""
                    i = 0
                    if "ASC" in tsql_string.upper():
                        tsql_string = (
                            tsql_string.replace("ASC", "")
                            .replace("asc", "")
                            .replace("ORDER BY", "ORDER BY ASC")
                        )
                    if "DESC" in tsql_string.upper():
                        tsql_string = (
                            tsql_string.replace("DESC", "")
                            .replace("desc", "")
                            .replace("ORDER BY", "ORDER BY DESC")
                        )
                case "GROUPBY":
                    subtree = self.parser(tsql_string)
                    children = self.find_children(subtree, "ColumnNameListContext")
                    tsql_string, aggs_in_query1 = self.aggregate_children(
                        tsql_string, children, aggs_in_query
                    )
                    if aggs_in_query1 == aggs_in_query:
                        aggs_in_query.append("COUNT(*)")
                    else:
                        aggs_in_query = aggs_in_query1
                case "PROJECT":
                    subtree = self.parser(tsql_string)
                    children = self.find_children(subtree, "ColumnNameListContext")
                    tsql_string, aggs_in_query = self.aggregate_children(
                        tsql_string, children, aggs_in_query
                    )
            item[1] = tsql_string
            if item[2]:
                for subq in item[2]:
                    if item[0][:-7] == "SETOPERATIONS":
                        item[1] = item[1].replace(
                            "SUBQUERY", self.stack_to_string(subq), 1
                        )
                    else:
                        item[1] = item[1].replace(
                            "SUBQUERY", self.stack_to_string(subq), 1
                        )
        if len(aggs_in_query) > 0:
            name = "AGGREGATE "
            compop = [">", "<", ">=", "<=", "=", "<>", "LIKE", "NOT", "IN"]
            for i, agg in enumerate(aggs_in_query):
                words = agg.split()
                result_words = []
                for word in words:
                    if word in compop:
                        break
                    result_words.append(word)
                aggs_in_query[i] = " ".join(result_words)
            for agg in list(set(aggs_in_query)):
                name += agg + ", "
            tsql_stack.append(["AGGREGATEContext", name[:-2], None])
        tsql_stack = self.reorder_stack(tsql_stack)
        query = ""

        if sum(node[0] == "JOINContext" for node in tsql_stack) >= 2:
            i = 0
            join_list = []
            while i < len(tsql_stack):
                item = tsql_stack[i]
                if tsql_stack[i][0][:-7] == "JOIN":
                    join_list.append(tsql_stack[i][1])
                    tsql_stack.pop(i)
                else:
                    if len(join_list) > 0:
                        break
                    i += 1
            for i, join in enumerate(join_list):
                p1, p2 = join.split(" WITH ")
                p2, p3 = p2.split(" ON ")
                p1 = p1.replace("FROM JOIN ", " ")
                p2 = p2.replace("FROM JOIN ", " ")
                join_list[i] = [p1, p2, p3]
            join_list = self.merge_similar_triplets(join_list)
            join_list = self.chain_triplets(join_list)
            join_query = "FROM JOIN " + self.triplets_to_string(join_list)
            tsql_stack.append(["JoinContext", join_query[5:], None])

        i = 0
        while i < len(tsql_stack):
            item = tsql_stack[i]
            if (
                i + 1 < len(tsql_stack)
                and "FROM" not in item[1].split(" ")[-2:]
                and "FROM" not in tsql_stack[i + 1][1].split(" ")[0]
                and tsql_stack[i + 1][0][:-7] != "SETOPERATIONS"
                and tsql_stack[i + 1][0][:-7] != "GROUPBY"
            ):
                query += item[1] + " FROM "
            else:
                if (
                    i + 1 < len(tsql_stack)
                    and "FROM" in item[1].split(" ")[-2:]
                    and "FROM" in tsql_stack[i + 1][1].split(" ")[0]
                ):
                    block = item[1].strip().split(" ")[:-1]
                    query += " ".join(block) + " "
                else:
                    query += item[1] + " "
            i += 1
        pattern = r"DISTINCT\s*\(([^)]+)\)"
        query = re.sub(pattern, lambda m: f"DISTINCT {m.group(1).strip()}", query)
        return query

    def print_tree(self, tree):
        for node in tree:
            indent = "  " * node["depth"]
            print(f"{indent}{node['type']}{node['name']}")

    def parser(self, tsql):
        """
        Converts TSQL string into parse tree using ANTLR4 grammar.
        
        Args: tsql (str): TSQL query string
        Returns: list: Tree of dictionaries containing node type, name, depth and parent index
        """
        lexer = newTSQLLexer(InputStream(tsql))
        lexer.removeErrorListeners()
        stream = CommonTokenStream(lexer)
        parser = newTSQLParser(stream)
        parser.removeErrorListeners()
        tree = parser.q()
        search_tree = self.build_tree(tree)
        return search_tree

    def tree_to_query(self, search_tree):
        """
        Converts a TSQL parse tree into a complete TSQL query string.
        
        Args: search_tree (list): List of dictionaries containing parsed TSQL nodes
        Returns: str: Final TSQL query with proper ordering
        """
        search_tree = self.clean_tree(search_tree)
        search_tree = self.clean_nodes(search_tree)
        # print_tree(search_tree)  # Use When Debugging
        tsql_stack = self.build_stack(search_tree)[0]
        query = self.stack_to_string(tsql_stack)
        return query