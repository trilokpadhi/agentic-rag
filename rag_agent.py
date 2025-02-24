# from langchain.retrievers import WikipediaRetriever
from langchain_community.retrievers import WikipediaRetriever # langchain.retrievers is deppreciated. 
# from langchain.chat_models import ChatOpenAI 
from langchain_community.chat_models import ChatOpenAI # langchain.chat_models is deppreciated. 
import os

class RAGAgent:
    def __init__(self, api_key):
        """Initialize the RAG agent with necessary components."""
        self.retriever = WikipediaRetriever()
        self.llm = ChatOpenAI(model="gpt-4o-2024-08-06", api_key=api_key)
 
 
    def refine_query(self, query, additional_context=""):
        prompt = (
            "Optimize the following query for better retrieval from Wikipedia, considering the additional context. "
            "Output only the refined query without any additional text or prefixes like 'Optimized Query:'.\n\n"
            f"Query: {query}\n\nAdditional Context: {additional_context}"
        )
        response = self.llm.invoke([
            {"role": "system", "content": "You are a query refiner."},
            {"role": "user", "content": prompt}
        ])
        return response.content.strip()  # Strip any accidental whitespace
    
    def create_unified_query(self, text_query, image_description, graph_description):
        combined = " ".join(filter(None, [text_query, image_description, graph_description]))
        prompt = (
            "Create a single, comprehensive query for Wikipedia retrieval based on the following information. "
            "Output only the query without any additional text or explanations.\n\n"
            f"Information: {combined}"
        )
        response = self.llm.invoke([
            {"role": "system", "content": "You are a query formulator."},
            {"role": "user", "content": prompt}
        ])
        return response.content.strip()


    def create_unified_query(self, text_query, image_description, graph_description):
        """
        Create a single query combining all modalities.
        
        Args:
            text_query (str): The text query.
            image_description (str, optional): Image description.
            graph_description (str, optional): Graph description.
        
        Returns:
            str: A unified, refined query.
        """
        combined = " ".join(filter(None, [text_query, image_description, graph_description]))
        prompt = f"Create a single, comprehensive query for Wikipedia retrieval based on: {combined}"
        response = self.llm.invoke([
            {"role": "system", "content": "You are a query formulator."},
            {"role": "user", "content": prompt}
        ])
        return response.content


    def generate_response(self, query, documents, additional_context):
        """
        Generate a response by summarizing documents and using additional context.
        
        Args:
            query (str): The original query.
            documents (list): List of retrieved documents.
            additional_context (str): Additional context from query descriptions.
        
        Returns:
            str: The generated response.
        """
        # Summarize documents
        summaries = []
        for doc in documents:
            summary_prompt = f"Summarize the following document in relation to the query: {query}\n\nDocument: {doc.page_content}"
            summary = self.llm.invoke([
                {"role": "system", "content": "You are a summarizer."},
                {"role": "user", "content": summary_prompt}
            ]).content
            summaries.append(summary)
        
        combined_summaries = "\n\n".join(summaries)
        prompt = f"Based on the following summaries and additional context, answer the query: {query}\n\nAdditional Context: {additional_context}\n\nSummaries:\n{combined_summaries}"
        response = self.llm.invoke([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
        return response.content


    def retrieve_documents(self, text_query=None, image_description=None, graph_description=None):
        all_docs = []
        full_context = " ".join(filter(None, [text_query, image_description, graph_description]))
        
        for query, modality in [
            (text_query, "text"),
            (image_description, "image"),
            (graph_description, "graph")
        ]:
            if query:
                try:
                    refined_query = self.refine_query(query, full_context)
                    docs = self.retriever.invoke(refined_query)
                    if not docs:
                        print(f"Warning: No documents retrieved for {modality} query: {refined_query}")
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Error retrieving documents for {modality} query: {e}")
        
        # Unified query
        try:
            unified_query = self.create_unified_query(text_query, image_description, graph_description)
            docs = self.retriever.get_relevant_documents(unified_query)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error retrieving documents for unified query: {e}")
        
        # Remove duplicates
        unique_titles = set()
        unique_docs = []
        for doc in all_docs:
            title = doc.metadata.get('title', '')
            if title not in unique_titles:
                unique_titles.add(title)
                unique_docs.append(doc)
        
        return unique_docs

    def process_query(self, text_query=None, image_description=None, graph_description=None):
        docs = self.retrieve_documents(text_query, image_description, graph_description)
        descriptions = " ".join(filter(None, [text_query, image_description, graph_description]))
        
        if not docs:
            return "Sorry, I couldn’t find enough information to answer your query."
        
        response = self.generate_response(text_query or "General query", docs, descriptions)
        return response

if __name__=='__main__':
    # Initialize agent
    api_key_tp = os.getenv("OPENAI_API_KEY_TP")
    agent = RAGAgent(api_key=api_key_tp)
    
    # Test the retriever
    test_docs = agent.retriever.invoke("Eiffel Tower") # changed from .get_relevant-documents to invoke() as the former is deppreciatwd. 
    print(f"Test retrieval for 'Eiffel Tower': {len(test_docs)} documents retrieved")

    # Sample queries
    text_query = "Tell me about the Eiffel Tower."
    image_description = "A tall iron tower in Paris."
    graph_description = "Eiffel Tower - located in - Paris; Eiffel Tower - built in - 1889."

    # Process with all modalities
    response = agent.process_query(text_query, image_description, graph_description)
    print("Response:", response)