from google.cloud import bigquery
import re
import json
import vertexai
import pandas
import pprint
from flask import jsonify
from vertexai.preview.language_models import ChatModel, InputOutputTextPair


def query_with_mappings(sql_query):
    bigquery_client = bigquery.Client()
    query_job = bigquery_client.query(sql_query)
    dataframe = query_job.to_dataframe()
    output_str = ""
    for idx in range(len(dataframe)):
        for head in dataframe.head():
            key, value = head, dataframe[head][idx]
            if dataframe[head][idx] is not None:
                formatted_key = str(key).replace("_", " ").title()
                output_str += f"{formatted_key} is {str(value)}\n"
        output_str += "\n"
    return output_str


def predict_large_language_model_sample(
    input_text: str,
    project_id: str,
    model_name: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    top_k: int,
    location: str = "us-central1"
):
    vertexai.init(project=project_id, location=location)

    chat_model = ChatModel.from_pretrained(model_name)
    parameters = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "top_p": top_p,
        "top_k": top_k,
    }

    chat = chat_model.start_chat(examples=[])
    response = chat.send_message(input_text, **parameters)
    return response.text


def formatResponseForDialogflow(texts, sessionInfo=None, targetFlow=None, targetPage=None):
    messages = []

    for text in texts:
        if text == "<None>":
            text = "None"
        messages.append(
            {
                "text": {"text": [text], "redactedText": [text]},
                "responseType": "HANDLER_PROMPT",
                "source": "VIRTUAL_AGENT",
            }
        )

    ret = {"fulfillment_response": {"messages": messages}}
    if sessionInfo:
        ret["sessionInfo"] = sessionInfo
    if targetFlow:
        ret["targetFlow"] = targetFlow
    if targetPage:
        ret["targetPage"] = targetPage
    return ret


def send_Json_response(concat_res):
    return jsonify(concat_res)


def convert_sql_query(query):
    pattern = r"(\w+)\s*=\s*'([^']*?)'"

    if 'WHERE' in query.upper():
        modified_query = re.sub(pattern, r"lower(\1) = '\2'", query, flags=re.IGNORECASE)
    else:
        modified_query = query

    return modified_query.replace("\\'", "'")


cache = {}


def hello_world(request):
    request_json = request.get_json(silent=True)

    if request_json and 'text' in request_json:
        text = request_json['text']

        table_Name = "supply-chain-twin-349311.SLT_L2_Canonical.orderSN"

        if table_Name in cache:
            cols = cache[table_Name]
        else:
            bigquery_client = bigquery.Client()
            cols = [col.name for col in bigquery_client.get_table(table_Name).schema]
            cache[table_Name] = cols

        input_text = f"""
        Question: {text}
        Table: {table_Name}
        Columns: {", ".join(cols)}
        Answer: bigquery sql query
        """

        model_output = predict_large_language_model_sample(
            input_text,
            "supply-chain-twin-349311",
            "chat-bison@001",
            0.2,
            1000,
            0.8,
            40,
            "us-central1"
        )

        regex = r"```sql((.|\n)*?)```"
        match = re.findall(regex, model_output, re.MULTILINE)
        sql_query = ""
        if len(match) != 0:
            first_match = match[0]
            if len(first_match) > 0:
                sql_query = ''.join(map(str, first_match)).strip()

        modified_query = convert_sql_query(sql_query)

        data = query_with_mappings(modified_query)

        final_text = f"Summarize the output precisely without any additional information:\n{data}"

        final_model_output = predict_large_language_model_sample(
            final_text,
            "supply-chain-twin-349311",
            "chat-bison@001",
            0.2,
            1000,
            0.8,
            40,
            "us-central1"
        )

        res = formatResponseForDialogflow([final_model_output])

        return send_Json_response(res)

    else:
        error_message = "Invalid field"
        res = formatResponseForDialogflow([error_message])
        return send_Json_response(res)
