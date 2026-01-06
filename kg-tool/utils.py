from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import requests
import json
from config import (
    OLLAMA_URL,
    OLLAMA_HEADERS,
    LLAMA_31_70B_BASE_URL,
    COREML_PROXY_URL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_OPENAI_API_KEY
)

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    
    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    
    # pattern = r"(?:\{\s*|\*\*)\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)\s*(?:\}|\*\*)"

    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    
    if not relations:
        pattern = r"\*\*\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)\*\*"
        for match in re.finditer(pattern, string):
            relation = match.group("relation").strip()
            if ';' in relation:
                continue
            score = match.group("score")
            if not relation or not score:
                return False, "output uncompleted.."
            try:
                score = float(score)
            except ValueError:
                return False, "Invalid score"
            if relation in head_relations:
                relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
            else:
                relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations

def build_messages(prompt: str):
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

def post_ollama_request(messages, model):
    payload = {
        "model": model,
        "messages": messages,
    }
    response = requests.post(OLLAMA_URL, headers=OLLAMA_HEADERS, data=json.dumps(payload))
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="deepseek"):
    messages = build_messages(prompt)

    if "llama32-3b" in engine.lower():
        model_name = "llama3.2"
        print(f"[llama32-3b] messages: {messages}")
        result = post_ollama_request(messages, model_name)
        print("result:", result)
        return result

    elif "deepseek" in engine.lower():
        model_name = "deepseek-r1:70b"
        print(f"[deepseek] messages: {messages}")
        result = post_ollama_request(messages, model_name)
        print("result:", result)
        return result

    elif "llama31-70b" in engine.lower():
        client = openai.OpenAI(
            api_key=DEFAULT_OPENAI_API_KEY,
            base_url=LLAMA_31_70B_BASE_URL,
            default_headers={"x-foo": "true"}
        )
        model_name = "Meta-Llama-3.1-70B-Instruct"
        print(f"[llama31-70b] messages: {messages}")

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        result = response.choices[0].message.content
        print("result:", result)
        return result

    else:
        # 默认调用 CoreML Proxy Service
        # 如果希望使用真正的 API key，可以在此用 opeani_api_keys 或从 config 中读取
        client = openai.OpenAI(
            api_key=DEFAULT_OPENAI_API_KEY,
            base_url=COREML_PROXY_URL
        )
        model_name = "azure-gpt-4o"

        
        f = 0
        while f == 0:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            result = response.choices[0].message.content
            f = 1

        print("result:", result)
        return result

    
def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(string, entity_candidates):
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]
    if len(scores) == len(entity_candidates):
        return scores
    else:
        print("All entities are created equal.")
        return [1/len(entity_candidates)] * len(entity_candidates)
    

def save_2_jsonl(question, answer, cluster_chain_of_entities, file_name, context=[]):
    context = "\n".join(context)
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities, "sub-question": context}
    with open("SubQ_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")

    
def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""
    

def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


def generate_without_explored_paths(question, args):
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'cwq_test':
        with open('../data/cwq_test.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'cwq_100':
        with open('../data/cwq_100.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string