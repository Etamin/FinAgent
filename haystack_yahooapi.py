import re
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack import Pipeline
from haystack.components.routers import ConditionalRouter
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
from pdf_rag.pipelines.generate_answer import run_retriever, run_generator
import yfinance as yf
import ast 
import gradio as gr


#logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
#logging.getLogger("haystack").setLevel(logging.DEBUG)
#tracing.tracer.is_content_tracing_enabled = True 
#tracing.enable_tracing(LoggingTracer(tags_color_strings={"haystack.component.input": "\x1b[1;31m", "haystack.component.name": "\x1b[1;34m"}))

#read input question from bash
#if len(sys.argv) < 2:
#    print("Usage: python hstranslate.py \"Your question here\"")
#    sys.exit(1)
#input_text = sys.argv[1]

def main(input_text, _:None):
    input_text = f"{input_text}"

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

    lang = detect(input_text) 
    ##translation output
    #first check language. Run if it is not English. Otherwise use English output like it is. 
    if lang == 'en':
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
              #print(query)
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
                    Columns: {{columns}};"
                    Answer:"""

        sql_query = SQLQuery('bank_demo.db')

        sql_pipe= Pipeline()
        sql_pipe.add_component("sql_prompt", PromptBuilder(sql_prompt, required_variables=[]))
        sql_pipe.add_component("sqlllm", OllamaGenerator(model="gemma3:12b"))
        sql_pipe.add_component("sql_querier", sql_query)

        sql_pipe.connect("sql_prompt", "sqlllm")
        sql_pipe.connect("sqlllm.replies", "sql_querier.queries")

    ####TO DO: No fallback to RAG
        try:
            fullresult = sql_pipe.run({
            "sql_prompt": {"question": question, "columns": columns},})
            result= fullresult['sql_querier']['results']  
            if "none" in str(result).lower():
                result = "No Answer"
                #result = ragpipe(question) #If sql retrieves no answer, go to rag (questions to similar)
                #fullresult = "RAG"
        except PipelineRuntimeError as e: #if runtimeerror because no cql result, go to rag (questions to similar)
            print(f"SQL Pipeline failed with error: {e}") 
            result = "No Answer"
            #result = ragpipe(question)
            #fullresult = "RAG"
        return result, fullresult 


    ###API
    def apipipe(question):
        #api prompt
        api_prompt = """If the company name mentioned in the question: "{{question}}" is explicitly APPLE, GOOGLE, BNP BGL PARIBAS or ACCELOR MITTAL, select a ticker symbol from this: {{tickers}}. If none of Apple, Google, BNP BGL Paribas or Accelor Mittal is explicitely mentioned, choose "None" as ticker symbol.
        If a concrete time period is mentioned, convert it like this: e.g. 2 months to "2mo" or one year to "1y", otherwise insert "1d".

        Create a list of strings from it:
        ["ticker symbol", "period"]

        Make sure your response stricktly follows the format.
        Do not include anything else in your Answer."""


        tickers = "BNP.PA, GOOG, AAPL, MT, None"

        api_pipe = Pipeline()
        api_pipe.add_component("api_prompt", PromptBuilder(api_prompt, required_variables=["tickers", "question"]))
        api_pipe.add_component("apillm", OllamaGenerator(model="gemma3:12b"))
        api_pipe.connect("api_prompt", "apillm")

        result = api_pipe.run({
            "question": question,
            "tickers": tickers,
        })

        llm_reply = result["apillm"]["replies"][0]

        try:
            # Use ast.literal_eval to safely parse the list
            api_prompt_result = ast.literal_eval(llm_reply)
            ticker_symbol, period = api_prompt_result[0], api_prompt_result[1]



            # Get stock data using yfinance
            ticker = yf.Ticker(ticker_symbol)
            time.sleep(1)
            data = ticker.history(period=period)
            time.sleep(1)

            fullapiresult = f"API Call: yf.Ticker('{ticker_symbol}').history(period='{period}')" + f"\n\n" + f"GET REQUEST: https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?range={period}&interval=1d"


            close_value = data.loc[data.index[0], "Close"]
            latest_date = data.index[0]
            latest_date_str = latest_date.date().isoformat()
            result = f"Current value: {close_value} on date: {latest_date_str}"
            return result, fullapiresult

        except Exception as e:
            result = "No Answer"
            fullapiresult = "Broken Api Call"
            return result, fullapiresult
            #print("Error in API pipeline", e)


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
        top_k_r = 3
        embedder_name = embedders_mapping['gte-base']
        llm = 'gemma3:12b'
        # User question
        query = question
        # Run retriever
        contexts = run_retriever(query, embedder_name, top_k, top_k_r)
        # Run generator
        clean_answer, metadata = run_generator(query, contexts, llm)
        
        return clean_answer, metadata

    def nl_answer(question, result, lang):
        ##Answer in Natural Language
        nllm = OllamaGenerator(model="gemma3:12b")
        out_prompt= PromptBuilder(template="""Based on question: '{{question}}' provide the result:{{query_result}}. Build an answer in normal language containing both. Do not provide extra Information (like language used) or explanations, just the plain Answer. Explain not, what any model does or did. Provide in English and a second time in language: {{lang}}. If there is 'None', 'No Answer' or an empty string in the result, just answer 'No Answer.'""", required_variables=[])
        answer_pipeline = Pipeline()
        answer_pipeline.add_component("out_prompt", out_prompt)
        answer_pipeline.add_component("nllm", nllm)
        answer_pipeline.connect("out_prompt", "nllm")
        input_data = {
            "out_prompt": {
                "query_result": result,
                "question": question,
                "lang":lang
            }
        }
        outresult = answer_pipeline.run(input_data)
        replies = outresult['nllm']['replies']
        result = f""+ replies[0] + "\n\n"
        return result

    #question = translate_to_english(input_text)
    question, direction = classify_direction(question)
    direction = direction.strip().lower()

    ###MAIN ACTION

    ###TO DO: remove RAG fallback
    if "sql" in direction:
        result, fullresult = sqlpipe(question)
        #if fullresult == "RAG":
        #    metadat = "- NOT IMPLEMENTED YET -"
        #else:
        metadat = fullresult['sql_querier']['queries']
        metadat=metadat[0]
        metadat=metadat.replace("```sql","").replace("```","")
        metadat=metadat.replace("sql","").replace("","")

    elif "api" in direction:
        result, metadata = apipipe(question)

    else:
        clean_answer, metadata = ragpipe(question)

    final_output = clean_answer

    if metadata is not None:
        filtered_meta = {k: v for k, v in metadata.items() if k != "all_metadata"}

        # Build the lines
        metadata_lines = "\n".join(f"{k}: {v}" for k, v in filtered_meta.items())

        # Then assemble final_output1
        final_output1 = f"{final_output}\n\nAnswer query/source is:\n{metadata_lines}\n"

        return final_output1
    else:
        # If metadata is None, just return the final_output
        return final_output

theme = gr.themes.Soft().set(
    block_label_background_fill="*primary_50",
    block_title_text_color='#636f85'
)

# Markdown content for description
markdown_description = """
### Welcome to our Partnershipday Demo!\n
You can do a lot of great stuff with it!\n
*Blablsbla \n
This is placed above.
"""

interface = gr.Interface(
    fn=main,
    inputs=[gr.Textbox(label="Your Question"),gr.Markdown("\n\n### Additional Info\nThis appears below")], #Also a way to include text
    outputs=gr.Textbox(label="Result"),
    title="Partnershipday Demo",
    description=markdown_description, 
    theme="base",
    flagging_mode="never",
)


interface.launch(share=True)
