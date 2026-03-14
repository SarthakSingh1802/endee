class QueryAgent:
    """
    Agent to process and refine user queries.
    """
    def __init__(self):
        pass

    def process_query(self, query):
        """
        Process the user query (e.g., strip whitespace, normalize).
        :param query: Raw user query string.
        :return: Processed query string.
        """
        return query.strip().lower()