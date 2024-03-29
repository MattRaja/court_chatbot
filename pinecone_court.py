import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.cohere import Cohere
from pinecone import Pinecone, PodSpec
from langchain.vectorstores.pinecone import Pinecone as PineconeVs
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import json
import re
import torch.cuda
import streamlit as st


class PineconeOperations:

    def __init__(self, api_key="6b193ec5-4353-4460-89f5-efcdc164bbcc"):
        self.index = None
        self.pc = Pinecone(api_key=api_key)
        self.create_index()
        self.connect_index()

    def create_index(self, index_name="court"):
        spec = PodSpec(environment="gcp-starter")

        indexes = self.pc.list_indexes()

        # create index if there are no indexes found
        if not len(indexes):
            self.pc.create_index(index_name, dimension=384, metric='cosine', spec=spec)

        return indexes

    def connect_index(self):
        indexes = self.create_index()
        # connect to a specific index
        self.index = self.pc.Index(indexes[0]["name"])

    def fetch_stats(self):
        # fetches stats about the index
        stats = self.index.describe_index_stats()
        return str(stats)

    def upsert(self, data):

        return json.loads(str(self.index.upsert(vectors=data)).replace("'", '"'))

    def query(self, query_vector):
        response = self.index.query(vector=query_vector.tolist(), top_k=1, include_metadata=True)
        return response

    def query_facts(self, query_vector):
        response = self.query(query_vector)

        return response.to_dict()['matches'][0]['metadata']['facts']

    def build_indexes(self, data, model):
        batch_size = 100

        for i in tqdm(range(0, len(data), batch_size)):
            # get end of batch
            i_end = min(len(data), i + batch_size)
            batch = data.iloc[i:i_end]
            # get metadata fields for this record
            metadatas = [{
                'name': record['name'],
                'facts': re.sub(r'<p>|</p>|</em>|\\n', '', record['facts']),
                'year': record['term'],
                'issue_area': record['issue_area']
            } for j, record in batch.iterrows()]
            # get the list of contexts / documents
            documents = list(zip(batch['facts'], batch['issue_area']))
            # create document embeddings
            embeds = model.vectorize_query(documents)
            # get IDs
            ids = batch['ID']
            # add everything to pinecone
            self.index.upsert(vectors=zip(ids, embeds, metadatas))


class Model:

    def __init__(self, model='all-MiniLM-L6-v2'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model, device=device)

    def vectorize_query(self, query):
        vector = self.model.encode(query)

        return vector


def get_data(path):
    df = pd.read_csv(path)

    df = pd.DataFrame(df)
    df.drop_duplicates(keep='first', inplace=True)

    # Convert ids from integers to strings
    df['ID'] = df['ID'].apply(str)
    # Replace empty fields with Unspecified
    df.fillna({'issue_area': 'Unspecified'}, inplace=True)

    return df


class Streamlit:

    def __init__(self, model, pc):

        self.model = model
        self.pc = pc
        st.subheader('Court Case Info')
        self.query = st.text_input("Enter your query")
        self.run_app()

    def run_app(self):
        if st.button("Submit"):
            # Validate inputs
            if not self.query:
                st.warning(f"Please enter a query.")
            else:
                try:
                    embeds = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
                    vectorstore = PineconeVs(self.pc.index, embeds, "facts")
                    # Initialize the OpenAI module, load and run the Retrieval Q&A chain
                    llm = Cohere(model="command-light", cohere_api_key="MAZCDW0GYy8veCZxvjKkSbxFUt2ZVpH6RLlypjVq")
                    qa = RetrievalQA.from_chain_type(llm,
                                                     chain_type="stuff",
                                                     retriever=vectorstore.as_retriever(search_kwargs={"k": 1}),
                                                     return_source_documents=True)
                    response = qa.invoke(self.query)
                    structured_response = self.structure_response(response)
                    st.success(structured_response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    def structure_response(self, response):
        sources = response['source_documents'][0].dict()
        name = sources['metadata']['name']
        year = sources['metadata']['year']
        result = response['result']
        source = sources['page_content']

        prompt_template = PromptTemplate.from_template("""
        AI: {result} 
        
        Source: {source}
        
        These are details from a case {name} in {year}
        """)

        prompt = prompt_template.format(result=result, source=source, name=name, year=year)
        return prompt


def main():
    model = Model()
    pc = PineconeOperations()
    # df = get_data('data/justice.csv')
    # pc.build_indexes(df, model)

    # query = ["Tell me about the most influential abortion case"]
    # response = pc.query_facts(model.vectorize_query(query))
    # print(response)

    app = Streamlit(model, pc)


if __name__ == '__main__':
    main()
