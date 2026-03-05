from retrieval.base_retriever import BaseRetriever
from retrieval.versionrag_retriever_db import VersionRAGRetrieverDatabase
from retrieval.versionrag_retriever_parser import VersionRAGRetrieverParser

class VersionRAGRetriever(BaseRetriever):
    def __init__(self):
        self.database = VersionRAGRetrieverDatabase()
        self.parser = VersionRAGRetrieverParser(self.database)
        super().__init__()
        
    def retrieve(self, query: str):
        retrieval_param = self.parser.parse_retrieval_mode(query=query)
        retrieval = self.database.retrieve(params=retrieval_param)
        return retrieval