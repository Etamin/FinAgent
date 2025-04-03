from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack import Pipeline
from haystack_integrations.components.retrievers import PDFRetriever
from haystack_integrations.components.sql import SQLQueryComponent
from haystack_integrations.components.api_caller import APICaller
from haystack.components.routers import ConditionalRouter
#from haystack.document_stores import FAISSDocumentStore
#from haystack.nodes import RAGenerator, DensePassageRetriever
#from haystack import Document
from typing import List
from haystack import component
from langdetect import detect
import pandas as pd
import sys
import sqlite3
import requests
import json

##read input question from bash
if len(sys.argv) < 2:
    print("Usage: python hstranslate.py \"Your question here\"")
    sys.exit(1)

input_text = sys.argv[1]

###TRANSLATION
##translation templates
template_de = "Translate {{query}} to German. Just answer with the translated text." #without pivoting, comment out
template_en = "Translate {{query}} to English. Just answer with the translated text."

pipe = Pipeline()

##translation components
pipe.add_component("prompt_builder", PromptBuilder(template=template_de, required_variables=[]))#without pivoting, comment out
pipe.add_component("llm", OllamaGenerator(model="gemma3:12b"))#without pivoting, comment out
pipe.connect("prompt_builder", "llm")#without pivoting, comment out

pipe.add_component("prompt_builder1", PromptBuilder(template=template_en, required_variables=[]))
pipe.add_component("llm1", OllamaGenerator(model="gemma3:12b"))
pipe.connect("prompt_builder1", "llm1")

##translation output
#first check language. Run if it is not English. Otherwise use English output like it is. 
if detect(input_text) == 'en':
    question = input_text
else:
    output = pipe.run({"prompt_builder":{"query": input_text}})#without pivoting, comment out
    output1 = output["llm"]["replies"][0]#without pivoting, comment out
    question = pipe.run({"prompt_builder1":{"query": output1}})#without pivoting, change output1 to input_text
    print(question["llm1"]["replies"][0])


###SQL Query
@component
class SQLQuery:

    def __init__(self, sql_database: str):
      self.connection = sqlite3.connect(sql_database, check_same_thread=False)

    @component.output_types(results=List[str], queries=List[str])
    def run(self, queries: List[str]):
        results = []
        for query in queries:
          query=query.replace("```sql","").replace("```","")
          result = pd.read_sql(query, self.connection)
          results.append(f"{result}")
        return {"results": results, "queries": queries}
    

##prompt to gen SQL Query

# Define database schema, [INSERT] if we use another one
transaction_columns = """
Transaction_ID VARCHAR(40) not null primary key,
Time INT,
Client_ID VARCHAR(12) references Source,
Beneficiary_ID VARCHAR(16) references Beneficiary,
Amount Float,
Currency VARCHAR(3),
Transaction_Type VARCHAR(20) not null
"""

# Modify table name if needed [INSERT]
sql_prompt = """Please generate an SQL query. The query should answer the following Question: {{question}};
            The query is to be answered for the table is called 'Transactions' with the following  
            Columns: {{columns}};
            Answer should only include a SQL query string without format like ```sql, no explnations or notes. You should only do question to SQL query translation
            Answer starts with "SELECT":"""


sql_query = SQLQuery('[INSERT].db')


##API
@component
class RESTCall:
        
    @component.output_types(results=str, queries=List[str])
    def run(self, queries:List[str]):
        print(queries[0])
        call=json.loads(queries[0])
        
        response = requests.get("[INSERT]"+call["api"], params=call["parameters"])
        return {"results": response.content.decode("utf-8"), "queries": queries}

  
#api prompt
api_prompt = """Please select an API function calling. The calling should answer the following Question: {{question}};
The query is to be related to a bank system and include these APIs with descriptions:
APIs: {{apis}};
===============
Make sure your response is only a simple string of API name.
Answer:"""

# Prompt template for generating API requests
call_prompt = """Please generate an API call. The call should answer the following Question: {{question}};
Use this:
API Name: {{api_name}}
Parameters: {{apipara}};

Use the following format:
{{apiformat}}

Make sure your response is a JSON object string without any format like ```json.
Answer:"""

# Define API format template
api_format = """
{
   "api":"apiName",
   "parameters":{
      "parameter1":"value1",
      "parameter2":"value2"
   }
}
"""

# Define available APIs, [INSERT] if we use other ones
api_list = """[GetAccountIDbyName, GetAccountNamebyID, GetAccount]"""



##RAG  TO DO
#rag_retriever = PDFRetriever(document_store="[INSERT]")
#rag_prompt = """[INSERT]"""

#document store
#document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", return_embedding=True)
# Write documents to document store
#document_store.write_documents(documents)
# Add documents embeddings to index
#document_store.update_embeddings(retriever=retriever)

#retriever
#retriever = DensePassageRetriever(
#    document_store=document_store,
#    query_embedding_model="[INSERT]",
#    passage_embedding_model="[INSERT]",
#    use_gpu=True,
#    embed_title=True,
#)

#generator = RAGenerator(
#    model_name_or_path="[INSERT]",
#    use_gpu=True,
#    top_k=1,
#    max_length=200,
#    min_length=2,
#    embed_title=True,
#    num_beams=2,
#)

##ROUTER

# Prompt template for port (sql/api/rag) classification
prompt_template_router = """
Create a classification result from the question: {{question}}.

Only if the question is asking about how much did the account ID transfer, please answer "sql".
Only if the question is asking about what is an account's ID of a name, please answer "api".
Only if the question is asking about [INSERT] , please answer "rag".
Only if the question is asking about anything else, answer "scope".

Only use information that is present in the passage. 
Make sure your response is a simple string that only can be 'sql' or 'api' or "rag" or "scope". No explanation or notes.
Answer:
"""


routes = [
    {
        "condition": "{{'sql' in replies[0]}}",  # If router response contains "sql", route to SQL query execution
        "output": "{{question}}",
        "output_name": "goto_sql",
        "output_type": str,
    },

# SHOULD WE IMPLEMENT THE CHECKER (FALLBACK PROMPT) HERE??
    {
        "condition": "{{sql_result is not None and sql_result != ''}}", 
        "output": "{{sql_result}}",
        "output_name": "final_answer",
        "output_type": str,
    },

# Go to API, if no result from SQL. Make it more failsafe. DO WE NEED THAT??
    #{
    #    "condition": "{{sql_result is None or sql_result == ''}}", # after no result from sql, check api
    #    "output": "{{question}}",
    #    "output_name": "goto_api",
    #    "output_type": str,
    #},

# (without the previous block or fallback) If no result from sql, error message 
    {
        "condition": "{{sql_result is None or sql_result == ''}}",
        "output": "'Information not found in scope of this demo'",
        "output_name": "final_answer",
        "output_type": str,
    },


    {
        "condition": "{{'api' in replies[0]}}",  # If router response contains "api", route to API call execution
        "output": "{{question}}",
        "output_name": "goto_api",
        "output_type": str,
    },

# SHOULD WE IMPLEMENT A CHECKER (FALLBACK PROMPT) HERE??
    {
        "condition": "{{api_result is not None and api_result != ''}}",
        "output": "{{api_result}}",
        "output_name": "final_answer",
        "output_type": str,
    },

# Go to RAG, if no result from SQL and/or API. Make it more failsafe. DO WE NEED THAT??
    #{
    #    "condition": "{{api_result is None or api_result == ''}}",
    #    "output": "{{question}}",
    #    "output_name": "goto_rag",
    #    "output_type": str,
    #},

# (without the previous block or fallback) If no result from api, error message 
    {
        "condition": "{{api_result is None or api_result == ''}}",
        "output": "'Information not found in scope of this demo'",
        "output_name": "final_answer",
        "output_type": str,
    },


# SHOULD WE IMPLEMENT A CHECKER (FALLBACK PROMPT) HERE??
    {
        "condition": "{{rag_result is not None and rag_result != ''}}", # If router response contains "rag", route to rag pipeline
        "output": "{{rag_result}}",
        "output_name": "final_answer",
        "output_type": str,
    },

# (without fallback) If no result from rag, error message 
    {
        "condition": "{{rag_result is None or rag_result == ''}}",
        "output": "'Information not found in scope of this demo'", # answer is not found in any of the provided sources. Return error message
        "output_name": "final_answer",
        "output_type": str,
    },



    {
        "condition": "{{'scope' in replies[0]}}",  # If response contains "scope", return error message
        "output": "{{question}}",
        "output_name": "goto_fallback",
        "output_type": str,
    },

    #    {
    #    "condition": "{{'scope' in replies[0]}}",  # If response contains "scope", return error message
    #    "output": "{'Information not found in scope of this demo'}",
    #    "output_name": "final_answer",
    #    "output_type": str,
    #},

]

router = ConditionalRouter(routes)

# Initialize SQLQuery and RESTCall components
sql_query = SQLQuery('[INSERT]')  # Connect to the database
api_caller = RESTCall()  # Create API calling component

##FALLBACK
fallback_prompt="""User entered a query that cannot be answerwed with the given information.
                                            The query was: {{question}}.
                                            Please try to answer the question:"""


##ROUTER PIPELINE
final_pipe = Pipeline()

# Add components to the pipeline
final_pipe.add_component("router", router)
final_pipe.add_component("router_prompt", PromptBuilder(prompt_template_router))
final_pipe.add_component("sql_prompt", PromptBuilder(sql_prompt))
final_pipe.add_component("api_prompt", PromptBuilder(api_prompt))
final_pipe.add_component("call_prompt", PromptBuilder(call_prompt))
final_pipe.add_component("fallback_prompt", PromptBuilder(fallback_prompt))

#final_pipe.add_component("rag_prompt", PromptBuilder(rag_prompt))

# Add LLM components (Ollama)
final_pipe.add_component("routerllm", OllamaGenerator(model="gemma3:12b"))
final_pipe.add_component("sqlllm", OllamaGenerator(model="gemma3:12b"))
final_pipe.add_component("apillm", OllamaGenerator(model="gemma3:12b"))
final_pipe.add_component("callllm", OllamaGenerator(model="gemma3:12b"))
final_pipe.add_component("fallback_llm", OllamaGenerator(model="gemma3:12b"))

#final_pipe.add_component("raglllm", OllamaGenerator(model="gemma3:12b"))

# Add SQL and API execution components
final_pipe.add_component("sql_querier", sql_query)
final_pipe.add_component("api_caller", api_caller)
#final_pipe.add_component("rag_retriever", rag_retriever)

# Connect pipeline components
final_pipe.connect("router_prompt", "routerllm")
final_pipe.connect("routerllm.replies", "router.replies")

final_pipe.connect("router.goto_sql", "sql_prompt.question")
final_pipe.connect("sql_prompt", "sqlllm")
final_pipe.connect("sqlllm.replies", "sql_querier")

final_pipe.connect("router.goto_api", "api_prompt.question")
final_pipe.connect("api_prompt", "apillm")
final_pipe.connect("apillm.replies", "call_prompt.api_name")
final_pipe.connect("call_prompt", "callllm")
final_pipe.connect("callllm.replies", "api_caller.queries")

final_pipe.connect("router.go_to_fallback", "fallback_prompt.question")
final_pipe.connect("fallback_prompt", "fallback_llm")

#final_pipe.connect("router.goto_rag", "rag_prompt.question") # the rag part of the pipeline relies on rag pipeline
#final_pipe.connect("rag_prompt", "ragllm")
#final_pipe.connect("ragllm.replies", "rag_prompt.rag_retriever")

# Run pipeline and print final_answer
result = final_pipe.run({
})
print(result.get("final_answer"))