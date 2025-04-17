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
from haystack import tracing
from haystack.tracing.logging_tracer import LoggingTracer
from haystack.core.errors import PipelineRuntimeError
import time
from pdf_rag.pipelines.generate_answer import run_retriever
from pdf_rag.pipelines.generate_answer import run_generator

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.DEBUG)
tracing.tracer.is_content_tracing_enabled = True 
tracing.enable_tracing(LoggingTracer(tags_color_strings={"haystack.component.input": "\x1b[1;31m", "haystack.component.name": "\x1b[1;34m"}))

#read input question from bash
#if len(sys.argv) < 2:
#    print("Usage: python hstranslate.py \"Your question here\"")
#    sys.exit(1)
#input_text = sys.argv[1]

input_text = "What is the account balance of account 12 am 30. September 2023?" 

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
    question = question["llm1"]["replies"][0]


def classify_direction(question):
    ###Decide which interface to use and output the direction
    routpipe = Pipeline()

    # Prompt template for port (sql/api/rag) classification [INSERT]
    #TO DO: BETTER PROMPT!!!
    routprompt = """
    Create a classification result from the question: {{question}}.

    Only if the question is about account balance (with account name), transactions (like transfer, money movements), payment, answer "sql".  Number can also refer to ID.
    Only if the question is about stock market, answer "api".
    Only if the question is about something else, answer "rag".

    Only use information that is present in the passage. 
    Make sure your response is a simple string that only can be 'sql' or 'api' or "rag". No explanation or notes.
    Answer:
    """

    routpipe.add_component("rout_prompt", PromptBuilder(template=routprompt, required_variables=[]))
    routpipe.add_component("routllm", OllamaGenerator(model="gemma3:12b"))

    routpipe.connect("rout_prompt", "routllm")

    direction = routpipe.run({"rout_prompt":{"question":question}})
    direction = direction["routllm"]["replies"][0]  
    return question, direction

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

def sqlpipe(question):
    ###SQL Query
    # Define database schema, [INSERT] if we use another one
    columns = "ROWID;transaction_id;account_id;date;amount;description;type"

    # Modify table name if needed [INSERT]
    sql_prompt = """Please generate an SQL query. The query should answer the following Question: {{question}};
                The query is to be answered for the table is called 'transactions' with the following
                Columns: {{columns}};
                Answer:"""
        
    sql_query = SQLQuery('bank_demo.db')

    sql_pipe= Pipeline()
    sql_pipe.add_component("sql_prompt", PromptBuilder(sql_prompt))
    sql_pipe.add_component("sqlllm", OllamaGenerator(model="gemma3:12b"))
    sql_pipe.add_component("sql_querier", sql_query)

    sql_pipe.connect("sql_prompt", "sqlllm")
    sql_pipe.connect("sqlllm.replies", "sql_querier.queries")
    
    try:
        result = sql_pipe.run({
        "sql_prompt": {"question": question, "columns": columns},})
        result= result['sql_querier']['results']  
        if "none" in str(result).lower():
            result = ragpipe(question) #If sql retrieves no answer, go to rag (questions to similar)
    except PipelineRuntimeError as e: #if runtimeerror because no cql result, go to rag (questions to similar)
        print(f"SQL Pipeline failed with error: {e}") 
        result = ragpipe(question)
    return result

###API
@component
class RESTCall:
        
    @component.output_types(results=str, queries=List[str])
    def run(self, queries:List[str]):
        print(queries[0])
        #queries = ["""
        #{
        #    "api_name": "GetAccount",
        #    "parameters": {
        #        "name": "alice dupont"
        #    }
        #}
        #"""]
        #print(queries[0])

        queries[0]=queries[0].replace("json","").replace("```","")
        call=json.loads(queries[0])


        #mapping from api_name to actual endpoint
        api_name_to_endpoint = {
            #"GetAccount": "/customers/",
            "GetAccountIDbyName": "/customers/id?name=",
            "GetAccountNamebyID": "/customers/name?id="
        #[INSERT] if there are more
        }

        #Map the API name to the actual API path
        api_path = api_name_to_endpoint.get(call["api_name"])
        if not api_path:
            raise ValueError(f"Unknown API name: {call['api_name']}")
        
        param_values = list(call["parameters"].values())
        if not param_values:
            raise ValueError("Missing parameter value")
        param_value = param_values[0]

        # Construct the final path
        full_path = api_path + str(param_value)

        # Send the GET request
        response = requests.get("http://localhost:3001" + full_path)   
        time.sleep(1)
        #response = requests.get("http://localhost:3001" + api_path, params=call["parameters"])
        return {"results": response.content.decode("utf-8"), "queries": queries[0]}

def apipipe(question):
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
    Api_name: {{api_name}}
    Parameters: {{apipara}};

    Use the following format:
    {{api_format}}

    Make sure your response is a JSON object string without any format like json. Parameters should just contain the parameter value. 
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
    api_list = """[GetAccountIDbyName, GetAccountNamebyID]"""

    transaction_columns="""
    Transaction_ID VARCHAR(40) not null primary key,
    Time INT,
    Client_ID VARCHAR(12) references Source,
    Beneficiary_ID VARCHAR(16) references Beneficiary,
    Amount Float,
    Currency VARCHAR(3),
    Transaction_Type VARCHAR(20) not null
    Source_Table VARCHAR(20)
    """

    api_list="""GetAccountIDbyName,
    GetAccountNamebyID
    """

    api_caller = RESTCall() 

    api_pipe= Pipeline()
    api_pipe.add_component("api_prompt", PromptBuilder(api_prompt))
    api_pipe.add_component("call_prompt", PromptBuilder(call_prompt))
    api_pipe.add_component("apillm", OllamaGenerator(model="gemma3:12b"))
    api_pipe.add_component("callllm", OllamaGenerator(model="gemma3:12b"))
    api_pipe.add_component("api_caller", api_caller)

    api_pipe.connect("api_prompt", "apillm")
    api_pipe.connect("apillm.replies", "call_prompt.api_name")
    api_pipe.connect("call_prompt", "callllm")
    api_pipe.connect("callllm.replies", "api_caller.queries")

    result = api_pipe.run({"question": question,
        "apiformat":api_format,
        "columns":transaction_columns,
        "apis":api_list,
        "apipara":"""{"name": string}"""
  })
    result= result['api_caller']['results']
    return result

def ragpipe(query):
    embedders_mapping = {
            'gte-base': 'Alibaba-NLP/gte-base-en-v1.5',
            'mutli-qa': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'bge-base-inst': 'BAAI/bge-base-en-v1.5',
            'alibaba-modern': 'Alibaba-NLP/gte-modernbert-base',
            'nomic-modern': 'nomic-ai/modernbert-embed-base',
            'modernbert-large': 'answerdotai/ModernBERT-large',
            'multi-qa-cos': 'sentence-transformers/multi-qa-mpnet-base-cos-v1'
    }
    
    # Parameters for rag 
    top_k = 30
    top_k_r = 5
    embedder_name = embedders_mapping['gte-base']
    llm = 'gemma3:12b'
    # User question
    query = question
    # Run retriever
    contexts = run_retriever(query, embedder_name, top_k, top_k_r)
    result = run_generator(query, contexts, llm)
    return result

def nl_answer(question, result):
    ##Answer in Natural Language
    nllm = OllamaGenerator(model="gemma3:12b")
    out_prompt= PromptBuilder(template="""Based on question: '{{question}}' provide the result:{{query_result}}. Build an answer in normal language containing both. Do not provide extra Information or explanations. Explain not, what any model does or did. Result:""")
    answer_pipeline = Pipeline()
    answer_pipeline.add_component("out_prompt", out_prompt)
    answer_pipeline.add_component("nllm", nllm)
    answer_pipeline.connect("out_prompt", "nllm")
    input_data = {
        "out_prompt": {
            "query_result": result,
            "question": question
        }
    }
    outresult = answer_pipeline.run(input_data)
    replies = outresult['nllm']['replies']
    print(f"\n\n\nAnswer to the Question is: "+ replies[0] + "\n\n\n")

#question = translate_to_english(input_text)
direction = classify_direction(question)


###MAIN ACTION
if "sql" in direction:
    result = sqlpipe(question)

elif "api" in direction:
    result = apipipe(question)

else:
    result = ragpipe(question)
    
final_output = nl_answer(question, result)
print(final_output)

