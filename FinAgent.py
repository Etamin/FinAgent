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
from datetime import datetime
from curl_cffi import requests

#logging
class DualStreamHandler:
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        if message != "\n":  # Avoid empty newline handling
            self.terminal.write(message)  # Output to terminal
            self.file.write(message)      # Output to file

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def isatty(self):
        return self.terminal.isatty()
    
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"/home/laura/PDay/FinAgent/logs/FinAgent_log_{timestamp}.log"
log_file = open(log_filename, "w")
logging.basicConfig(filename=log_filename, level=logging.DEBUG)
log_stream = open(log_filename, "a")  # Open in append mode
dual_handler = DualStreamHandler(sys.stdout, log_stream)
sys.stdout = dual_handler
sys.stderr = dual_handler
tracing.tracer.is_content_tracing_enabled = True
tracing.enable_tracing(LoggingTracer(tags_color_strings={ 
    "haystack.component.input": "\x1b[1;31m", 
    "haystack.component.name": "\x1b[1;34m"
}))

#wrapped in main function for gradio UI
def main(input_text, _=None):
    print("FinAgent is ready!")
    allstarttime = time.time()

    #time formatting
    def format_execution_time(start_time, end_time):
        execution_time = end_time - start_time
        hours, rem = divmod(execution_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_formatted = "{}h {}m {}s".format(int(hours), int(minutes), int(seconds))
        return time_formatted


    input_text = f"{input_text}"

    ###TRANSLATION
    start_time = time.time()
    ##translation templates
    langtemplate = "detect the language in the following sentence: {{query}}. Answer only with the ISO 639 language code."
    template_de = "Translate {{query}} to German. Just answer with the translated text."
    template_en = "Translate {{query}} to English. Just answer with the translated text."

    ##translation components
    pipe = Pipeline()
    pipe.add_component("prompt_builder_base", PromptBuilder(template=langtemplate, required_variables=[]))
    pipe.add_component("basellm", OllamaGenerator(model="gemma3:12b"))
    pipe.connect("prompt_builder_base", "basellm")

    pipe.add_component("prompt_builder", PromptBuilder(template=template_de, required_variables=[]))
    pipe.add_component("llm", OllamaGenerator(model="gemma3:12b"))
    pipe.connect("prompt_builder", "llm")


    pipe.add_component("prompt_builder1", PromptBuilder(template=template_en, required_variables=[]))
    pipe.add_component("llm1", OllamaGenerator(model="gemma3:12b"))
    pipe.connect("prompt_builder1", "llm1")

    
    #first check language
    lang1 = pipe.run({"prompt_builder_base":{"query": input_text}})
    lang = lang1["basellm"]["replies"][0]

    if lang == 'en': #keep as it is
        question = input_text
    elif lang == 'lb': #pivoting with german
        output = pipe.run({"prompt_builder":{"query": input_text}})
        output1 = output["llm"]["replies"][0]
        question = pipe.run({"prompt_builder1":{"query": output1}})
        question = question["llm1"]["replies"][0]
    else: #translate de and fr to en (other languages work too)
        output = pipe.run({"prompt_builder1":{"query": input_text}})
        question = output["llm1"]["replies"]

    end_time = time.time()   
    translation_time = end_time - start_time
    translation_time = format_execution_time(start_time, end_time) 
    print(f"\n\nTranslation process finished. Total Translation time: {translation_time}\n\n")

    ###ROUTING
    def classify_direction(question):
        ###Decide which interface to use and output the direction
        start_time = time.time()

        ##Prompt template for port (sql/api/rag) classification
        routprompt = """
        Create a classification result from the question: {{question}}.

        Only if the question is about account balance (with account name), transactions (like transfer, money movements), payment, answer "sql".  Number can also refer to ID.
        Only if the question is about stock market, answer "api".
        Only if the question is about AMCOR, Ulta Beauty, Foot Locker or something else, answer "rag".

        Only use information that is present in the passage. 
        Make sure your response is a simple string that only can be 'sql' or 'api' or "rag". No explanation or notes.
        Answer:
        """

        ##routing components
        routpipe = Pipeline()
        routpipe.add_component("rout_prompt", PromptBuilder(template=routprompt, required_variables=[]))
        routpipe.add_component("routllm", OllamaGenerator(model="gemma3:12b"))
        routpipe.connect("rout_prompt", "routllm")

        #get direction
        direction = routpipe.run({"rout_prompt":{"question":question}})
        direction = direction["routllm"]["replies"][0]  
        end_time = time.time()   
        routing_time = end_time - start_time
        routing_time = format_execution_time(start_time, end_time) 
        print(f"\n\nRouting process finished. Total Routing time: {routing_time}\n\n")
        return question, direction


    ###SQL
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
            time.sleep(1)
            self.connection.close() 
            return {"results": results, "queries": queries}

    def sqlpipe(question):
        start_time = time.time()

        ##columns found in the specified database [CUSTOM]
        columns = "ROWID;transaction_id;account_id;date;amount_eur;description;type"

        ##template prompt for sql query
        sql_prompt = """Please generate an SQL query. The query should answer the following Question: {{question}}; Use Currency from column names.
                    The query is to be answered for the table is called 'transactions' with the following
                    Columns: {{columns}};"
                    Answer:"""

        ##database path
        sql_query = SQLQuery('/home/laura/PDay/FinAgent/bank_demo.db')

        ##sql pipeline components
        sql_pipe= Pipeline()
        sql_pipe.add_component("sql_prompt", PromptBuilder(sql_prompt, required_variables=[]))
        sql_pipe.add_component("sqlllm", OllamaGenerator(model="gemma3:12b"))
        sql_pipe.add_component("sql_querier", sql_query)

        sql_pipe.connect("sql_prompt", "sqlllm")
        sql_pipe.connect("sqlllm.replies", "sql_querier.queries")

        ##SQL pipeline with fallback for error
        try:
            fullresult = sql_pipe.run({
            "sql_prompt": {"question": question, "columns": columns},})
            result= fullresult['sql_querier']['results'] 
            fullresult =fullresult['sql_querier']['queries'] 
            if "none" in str(result).lower():
                result = "No Answer"
        except PipelineRuntimeError as e: 
            print(f"\n\nSQL Pipeline failed with error: {e}\n\n") 
            result = "No Answer"
            fullresult = "No Answer"
        end_time = time.time()   
        sql_time = end_time - start_time
        sql_time = format_execution_time(start_time, end_time) 
        print(f"\n\nSQL process finished. Total SQL time: {sql_time}\n\n")
        return result, fullresult 


    ###API
    def apipipe(question):
        session = requests.Session(impersonate="chrome") #specified for used yfinance api
        start_time = time.time()

        ##api prompt template
        api_prompt = """If the company name mentioned in the question: "{{question}}" is explicitly APPLE, GOOGLE, BNP BGL PARIBAS or ACCELOR MITTAL, select a ticker symbol from this: {{tickers}}. If none of Apple, Google, BNP BGL Paribas or Accelor Mittal is explicitely mentioned, choose "None" as ticker symbol.
        If a concrete time period is mentioned, convert it like this: e.g. 2 months to "2mo" or one year to "1y", otherwise insert "1d".

        Create a list of strings from it:
        ["ticker symbol", "period"]

        Make sure your response stricktly follows the format.
        Do not include anything else in your Answer."""

        ##used ticker symbols [CUSTOM]
        tickers = "BNP.PA, GOOG, AAPL, MT, None"

        ##api pipeline components
        api_pipe = Pipeline()
        api_pipe.add_component("api_prompt", PromptBuilder(api_prompt, required_variables=["tickers", "question"]))
        api_pipe.add_component("apillm", OllamaGenerator(model="gemma3:12b"))
        api_pipe.connect("api_prompt", "apillm")

        ##api pipeline with fallback
        result = api_pipe.run({
            "question": question,
            "tickers": tickers,
        })

        llm_reply = result["apillm"]["replies"][0]

        try:
            api_prompt_result = ast.literal_eval(llm_reply)
            ticker_symbol, period = api_prompt_result[0], api_prompt_result[1]

            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?range={period}&interval=1d"
            response = session.get(url)
            time.sleep(1)
            data_json = response.json()
            result = data_json['chart']['result'][0] #get certain result from json
            timestamps = result['timestamp']
            indicators = result['indicators']['quote'][0]
            data = pd.DataFrame(indicators)
            data['timestamp'] = pd.to_datetime(timestamps, unit='s')
            data.set_index('timestamp', inplace=True)
            fullapiresult = (
                f"Answer found via API.\n"
                f"API Call: GET {url}\n\n"
            )
            close_value = data.loc[data.index[0], "close"]
            latest_date = data.index[0]
            latest_date_str = latest_date.date().isoformat()
            result = f"Current value: {close_value} on date: {latest_date_str}"
            end_time = time.time()   
            api_time = end_time - start_time
            api_time = format_execution_time(start_time, end_time) 
            print(f"\n\nAPI process finished. Total API time: {api_time}\n\n")
            return result, fullapiresult

        except Exception as e:
            result = "No Answer"
            fullapiresult = "Broken Api Call"
            return result, fullapiresult

    ###RAG
    def ragpipe(query):
        start_time = time.time()
        embedders_mapping = {
                'gte-base': 'Alibaba-NLP/gte-base-en-v1.5',
                'mutli-qa': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
                'bge-base-inst': 'BAAI/bge-base-en-v1.5',
                'alibaba-modern': 'Alibaba-NLP/gte-modernbert-base',
                'nomic-modern': 'nomic-ai/modernbert-embed-base',
                'modernbert-large': 'answerdotai/ModernBERT-large',
                'multi-qa-cos': 'sentence-transformers/multi-qa-mpnet-base-cos-v1'
        }

        ##Parameters for rag [CUSTOM]
        top_k = 30
        top_k_r = 5
        embedder_name = embedders_mapping['gte-base']
        llm = 'o4-mini'
        #Run retriever
        contexts = run_retriever(query, embedder_name, top_k, top_k_r)
        #Run generator
        clean_answer, metadata = run_generator(query, contexts, llm)

        end_time = time.time()   
        rag_time = end_time - start_time
        rag_time = format_execution_time(start_time, end_time) 
        print(f"\n\nRAG process finished. Total RAG time: {rag_time}\n\n")

        return clean_answer, metadata
    
    ###Answer in Natural Language (Output in origininal language + English)
    def nl_answer(question, result, lang):
        start_time = time.time()
        nllm = OllamaGenerator(model="gemma3:12b")
        out_prompt= PromptBuilder(template="""Based on question: '{{question}}' provide the result:{{query_result}}. Build an answer in normal language containing both. Do not provide extra Information (like language used) or explanations, just the plain Answer. Explain not, what any model does or did. Provide in English and a second time in the following language:{{lang}}. If there is 'None', 'No Answer' or an empty string in the result, just answer 'No Answer.'""", required_variables=[])
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
        end_time = time.time()   
        answer_time = end_time - start_time
        answer_time = format_execution_time(start_time, end_time) 
        print(f"\n\nGenerating Answer in Natural Language finished. Total Generation time: {answer_time}\n\n")
        return result


    ###MAIN ACTION
    question, direction = classify_direction(question)
    direction = direction.strip().lower()

    if "sql" in direction:
        result, metadata = sqlpipe(question)
        metadata=metadata[0]
        metadata=metadata.replace("```sql","").replace("```","")
        metadata=metadata.replace("sql","").replace("","")
        metadata=f"Answer found via SQL." + f"\n" + metadata

    elif "api" in direction:
        result, metadata = apipipe(question)

    elif "rag" in direction:
        result, metadata = ragpipe(question)
        final_output = result

        if metadata:
            # drop the “all_metadata” entry
            filtered_meta = {
                k: v for k, v in metadata.items()
                if k != "all_metadata"
            }
            metadata = "\n".join(f"{k}: {v}" for k, v in filtered_meta.items())


    final_output = nl_answer(question, result, lang)
    if metadata:
        final_output1 = final_output + "Answer query/source is:" + f"\n" + metadata + f"\n"
    else:
        final_output1 = final_output
    allend_time = time.time()   
    all_time = allend_time - allstarttime
    all_time = format_execution_time(allstarttime, allend_time) 
    print(f"\n\nThe whole process finished. Total process time: {all_time}\n\n")
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    return final_output1


###UI
theme = gr.themes.Base().set(
    body_text_color='black',
    background_fill_primary='white',
    block_background_fill='*primary_950',
    block_border_color='*primary_900',
    block_info_text_color='black',
    block_label_background_fill='*primary_50',
    block_title_text_color='black',
    input_background_fill='white'
)

def clear_inputs():
    return "", ""

with gr.Blocks(title="Partnership Day Demo", theme=theme) as demo:
    gr.Markdown("""
    ### Welcome to our Partnershipday Demo!
    What you see here is a limited demonstration of our "FinAgent" project, that we created in collaboration with BGL BNP Paribas.
    You can enter a simple question about data stored in our sql database, our pdfs or the stock market.
    You will receive the answer as well as the source of it. 
    """)

    gr.HTML("""
        <script>
            window.onload = function() {
                const input = document.querySelector('input[type="text"]');
                if (input) {
                    input.focus();
                }
            }
        </script>
    """)

    with gr.Row(): 
        with gr.Column():
            input_text = gr.Textbox(label = "", placeholder="Your Question")
            with gr.Row(): 
                clear_btn = gr.Button("Clear", variant="secondary")
                start_btn = gr.Button("Start", variant="primary")

        output_text = gr.Textbox(label="", placeholder= "Result") 

    #trigger main with both button and Enter key
    start_btn.click(fn=main, inputs=input_text, outputs=output_text)
    input_text.submit(fn=main, inputs=input_text, outputs=output_text)
    clear_btn.click(fn=clear_inputs, outputs=[input_text, output_text])

demo.launch(share=True, debug=True)