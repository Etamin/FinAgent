from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack import Pipeline
#from haystack_integrations.components.retrievers import PDFRetriever
#from haystack_integrations.components.sql import SQLQueryComponent
#from haystack_integrations.components.api_caller import APICaller
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
import logging
import logging
from haystack import tracing
from haystack.tracing.logging_tracer import LoggingTracer
from haystack.core.errors import PipelineRuntimeError

#logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
#logging.getLogger("haystack").setLevel(logging.DEBUG)
#
#tracing.tracer.is_content_tracing_enabled = True # to enable tracing/logging content (inputs/outputs)
#tracing.enable_tracing(LoggingTracer(tags_color_strings={"haystack.component.input": "\x1b[1;31m", "haystack.component.name": "\x1b[1;34m"}))

#read input question from bash
if len(sys.argv) < 2:
    print("Usage: python hstranslate.py \"Your question here\"")
    sys.exit(1)
input_text = sys.argv[1]

#input_text = 

###TRANSLATION
##translation templates
template_de = "Translate {{query}} to German. Just answer with the translated text. Dates should always follow the format YYYY-MM-DD." #without pivoting, comment out
template_en = "Translate {{query}} to English. Just answer with the translated text. Dates should always follow the format YYYY-MM-DD."

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
          query=query.replace("sql","").replace("","")
          result = pd.read_sql(query, self.connection)
          results.append(f"{result}")
        self.connection.close() 
        return {"results": results, "queries": queries}


# Define database schema, [INSERT] if we use another one
columns = "ROWID;transaction_id;account_id;date;amount;description;type"

# Modify table name if needed [INSERT]
sql_prompt = """Please generate an SQL query. The query should answer the following Question: {{question}};
            The query is to be answered for the table is called 'transactions' with the following
            Columns: {{columns}};
            Answer:"""

            
sql_query = SQLQuery('bank_demo.db')
llm = OllamaGenerator(model="gemma3:12b")


##API TO DO
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
{{api_format}}

Make sure your response is a JSON object string without any format like 
json.
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

api_caller = RESTCall() 

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

# Prompt template for port (sql/api/rag) classification [INSERT]
#TO DO: BETTER PROMPT!!!
prompt_template_router = """
Create a classification result from the question: {{question}}.

Only if the question is about account balance (with account name), transactions (like transfer, money movements), payment, answer "sql".
Only if the question is about an account name or profiles, answer "api".
Only if the question is about rag, answer "rag".

Only use information that is present in the passage. 
Make sure your response is a simple string that only can be 'sql' or 'api' or "rag". No explanation or notes.
Answer:
"""

routes = [
    {
        "condition": "{{'sql' in replies[0]}}",  # If router response contains "sql", route to SQL query execution
        "output": "{{question}}",
        "output_name": "goto_sql",
        "output_type": str,
    },

        {
        "condition": "{{'api' is in replies[0]}}",  
        "output": "'Tried an api call. Not implemented yet.'",
        "output_name": "answer",
        "output_type": str,
    },

    {
        "condition": "{{'sql' is not in replies[0] and 'api' is not in replies[0]}}",  
        "output": "'Tried an api call. Not implemented yet.'",  
        "output_name": "answer",
        "output_type": str,
    },

    {
        "condition": "{{replies[0] is not none and replies[0] != ''}}", 
        "output": "{{replies}}",
        "output_name": "answer",
        "output_type": List[str],
    },

    {
        "condition": "{{replies[0].text is none or replies[0] == ''}}",
        "output": "'Information not found in scope of this demo'",
        "output_name": "answer",
        "output_type": str,
    },



    #{
    #    "condition": "{{'api' in replies[0]}}",  # If router response contains "api", route to API call execution
    #    "output": "{{question}}",
    #    "output_name": "goto_api",
    #    "output_type": str,
    #},
#
    #{
    #    "condition": "{{replies[0].text is not none and replies[0].text != ''}}",
    #    "output": "{{replies[0].text}}",
    #    "output_name": "answer",
    #    "output_type": str,
    #},
#
    #{
    #    "condition": "{{replies[0].text is none or replies[0].text == ''}}",
    #    "output": "'Information not found in scope of this demo'",
    #    "output_name": "answer",
    #    "output_type": str,
    #},


#    {
#        "condition": "{{rag_result is not none and rag_result != ''}}", # If router response contains "rag", route to rag pipeline
#        "output": "{{rag_result}}",
#        "output_name": "final_answer",
#        "output_type": str,
#    },
#
#    {
#        "condition": "{{rag_result is none or rag_result == ''}}",
#        "output": "'Information not found in scope of this demo'", # answer is not found in any of the provided sources. Return error message
#        "output_name": "final_answer",
#        "output_type": str,
#    },
#
#    {
#        "condition": "{{'scope' in replies[0]}}",  # If response contains "scope", return error message
#        "output": "{{question}}",
#        "output_name": "goto_fallback",
#        "output_type": str,
#    },
#
]


##FALLBACK
#fallback_prompt="""User entered a query that cannot be answerwed with the given information.
#                                            The query was: {{question}}.
#                                            Please try to answer the question:"""


router = ConditionalRouter(routes)

final_pipe= Pipeline()
# Add basiccomponents to the pipeline
final_pipe.add_component("router", router)
final_pipe.add_component("router_prompt", PromptBuilder(prompt_template_router))
final_pipe.add_component("sql_prompt", PromptBuilder(sql_prompt))
#final_pipe.add_component("api_prompt", PromptBuilder(api_prompt))
#final_pipe.add_component("call_prompt", PromptBuilder(call_prompt))
#final_pipe.add_component("fallback_prompt", PromptBuilder(fallback_prompt)

# Add LLM components (Ollama)
final_pipe.add_component("routerllm", OllamaGenerator(model="gemma3:12b"))
final_pipe.add_component("sqlllm", OllamaGenerator(model="gemma3:12b"))
#final_pipe.add_component("apillm", OllamaGenerator(model="gemma3:12b"))
#final_pipe.add_component("callllm", OllamaGenerator(model="gemma3:12b"))
#final_pipe.add_component("fallback_llm", OllamaGenerator(model="gemma3:12b"))
#final_pipe.add_component("raglllm", OllamaGenerator(model="gemma3:12b"))

# Add SQL and API execution components
final_pipe.add_component("sql_querier", sql_query)
#final_pipe.add_component("api_caller", api_caller)
#final_pipe.add_component("rag_retriever", rag_retriever)


#Connect pipeline components
final_pipe.connect("router_prompt", "routerllm")
final_pipe.connect("routerllm.replies", "router.replies")


final_pipe.connect("router.goto_sql", "sql_prompt.question")
final_pipe.connect("sql_prompt", "sqlllm")
final_pipe.connect("sqlllm.replies", "sql_querier.queries")


#final_pipe.connect("router.goto_api", "api_prompt.question")
#final_pipe.connect("api_prompt", "apillm")
#final_pipe.connect("apillm.replies", "call_prompt.api_name")
#final_pipe.connect("call_prompt", "callllm")
#final_pipe.connect("callllm.replies", "api_caller.queries")

try:
    result = final_pipe.run({ "router":{"question": question}, "sql_prompt": {"columns": columns} })

    #result = final_pipe.run({ "router":{"question": question}, "sql_prompt": {"columns": columns}, "api_prompt": {"apis":api_list}, "call_prompt":{ "apipara":"parabla", "api_format": api_format} })

    query_result= result["sql_querier"]["results"][0]

    ##Answer in Natural Language
    nllm = OllamaGenerator(model="gemma3:12b")
    out_prompt= PromptBuilder(template="""Based on question: '{{question}}' provide the result:{{query_result}}. Build an answer in normal language containing both. Do not provide extra Information or explanations. Explain not, what any model does or did. Result:""")

    answer_pipeline = Pipeline()
    answer_pipeline.add_component("out_prompt", out_prompt)
    answer_pipeline.add_component("nllm", nllm)
    answer_pipeline.connect("out_prompt", "nllm")

    input_data = {
        "out_prompt": {
            "query_result": query_result,
            "question": question
        }
    }

    outresult = answer_pipeline.run(input_data)
    replies = outresult['nllm']['replies']

    sql_query_used = result["sql_querier"]["queries"][0]
    print(f"Executed SQL query was: \n\n "+ sql_query_used)
    print(f"\n\n\nAnswer to the Question is: "+ replies[0] + "\n\n\n")

except PipelineRuntimeError as e:
    print(f"SQL Query is malformed. Try again.")
