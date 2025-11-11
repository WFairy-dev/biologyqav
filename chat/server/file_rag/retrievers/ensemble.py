from __future__ import annotations

from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever

from chat.server.file_rag.retrievers.base import BaseRetrieverService

import spacy

# 加载 spaCy 的英文分词模型
# 请确保已安装 "en_core_web_sm" 模型，可以使用命令：python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def spacy_tokenize(text: str):
    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]

class EnsembleRetrieverService(BaseRetrieverService):
    def do_init(
        self,
        retriever: BaseRetriever = None,
        top_k: int = 5,
    ):
        self.vs = None
        self.top_k = top_k
        self.retriever = retriever

    @staticmethod
    def from_vectorstore(
        vectorstore: VectorStore,
        top_k: int,
        score_threshold: int | float,
    ):
        # 通过向量库构造基于向量相似度的检索器
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": top_k},
        )
        # 从向量库中提取原始文档，并使用 spaCy 进行英文分词，
        # 构造基于 BM25 的检索器
        docs = list(vectorstore.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(
            docs,
            preprocess_func=spacy_tokenize,
        )
        bm25_retriever.k = top_k
        # 组合两种检索器，权重均为 0.5
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        return EnsembleRetrieverService(retriever=ensemble_retriever, top_k=top_k)

    def get_relevant_documents(self, query: str):
        return self.retriever.get_relevant_documents(query)[: self.top_k]
