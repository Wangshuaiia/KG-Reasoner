from SPARQLWrapper import SPARQLWrapper, JSON
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

from gnn_retrieval import MyModel
from backtrack import reasoning_with_backtrack
from heapq import nlargest
from config import MODEL_CONFIG, SPARQL_CONFIG

device = MODEL_CONFIG["device"]

model_infer = MyModel(
    bert_name=MODEL_CONFIG["bert_name"],
    bert_hidden_dim=MODEL_CONFIG["bert_hidden_dim"],
    gnn_hidden_dim=MODEL_CONFIG["gnn_hidden_dim"],
    num_classes=MODEL_CONFIG["num_classes"]
)

model_infer.load_state_dict(torch.load(MODEL_CONFIG["model_state_path"], map_location=device))
model_infer.to(device)
model_infer.eval()

# SPARQL_CONFIG 
SPARQLPATH = SPARQL_CONFIG["SPARQLPATH"]
sparql_head_relations = SPARQL_CONFIG["sparql_head_relations"]
sparql_tail_relations = SPARQL_CONFIG["sparql_tail_relations"]
sparql_tail_entities_extract = SPARQL_CONFIG["sparql_tail_entities_extract"]
sparql_head_entities_extract = SPARQL_CONFIG["sparql_head_entities_extract"]
sparql_id = SPARQL_CONFIG["sparql_id"]


def gnn_based_selection_single(question, entity_id, relation):
    score = 0
    entity_candidates_id = entity_search(entity_id, relation, True)
    neighbors_texts = [id2entity_name_or_type(e_id) for e_id in entity_candidates_id]
    entity_texts = [id2entity_name_or_type(entity_id)]
    
    with torch.no_grad():
        logits = model_infer(entity_texts, [neighbors_texts], device=device)
        predictions = logits.argmax(dim=-1)
        score = predictions.cpu().tolist()[0]

    return score


def gnn_based_selection(question, entity_candidates_id, relation, width=3):
    score_list = []
    for entity_id in entity_candidates_id:
        score = gnn_based_selection_single(question, entity_id, relation)
        score_list.append(score)
    top_k = nlargest(width, zip(score_list, entity_candidates_id))  
    
    topn_entities = [entity for score, entity in top_k]
    topn_scores = [score for score, entity in top_k]
    
    return topn_entities, topn_scores


def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True


def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]


def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']
    
from freebase_func import *
from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

def construct_relation_prune_prompt_with_subquestion(original_question, entity_name, total_relations, sub_question):
    combined_prompt = relation_selection_prompt + '\n**Original Question:** ' + original_question + '\n*Topic Entity:** ' + entity_name + '\n**Subquestion:** ' + sub_question + '\n**Relations:**' + '; '.join(total_relations) + '\n\n**Output:**'
    return combined_prompt

def construct_question_gen_prompt(question, context=[]):
    context = '\n'.join(context).strip()
    if context:
        # has context
        combined_prompt = questionGen_prompt + '\n- **Original Question:** ' + question + '\n- **Previously Generated Sub-questions and Answers:**\n' + context + '\n**Output:**\n- **Next Sub-question:**'
    else:
        # no context
        combined_prompt = questionGen_prompt + '\n- **Original Question:** ' + question + '\n- **Previously Generated Sub-questions and Answers:** None \n' + '\n**Output:**\n- **Next Sub-question:**'
    return combined_prompt

def sub_question_gen(original_question, args, context=[]):
    question_gen_prompt = construct_question_gen_prompt(original_question, context)
    print("question_gen_prompt:", question_gen_prompt)
    sub_question = run_llm(question_gen_prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.question_LLM_type)
    return sub_question

def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
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


def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
    

def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '


def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, args, sub_question=''):
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    print("sparql_relations_extract_head:\n")
    print(sparql_relations_extract_head)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    
    sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations.sort()  
    
    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt_with_subquestion(question, entity_name, total_relations, sub_question)
        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        print("entity_id:", entity_id)
        print("head_relations:", head_relations)
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations)
        print("retrieve_relations_with_scores:", retrieve_relations_with_scores)

    elif args.prune_tools == "bm25":
        topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 

    if flag:
        return retrieve_relations_with_scores
    else:
        return [] 
    
    
def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (entity, relation)
        entities = execurte_sparql(head_entities_extract)


    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity


def entity_score(question, entity_candidates_id, score, relation, args, sub_question=""):
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
    if all_unknown_entity(entity_candidates):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    if args.prune_tools == "llm":
        if sub_question:
            prompt = construct_entity_score_prompt(sub_question, relation, entity_candidates)
        else:
            prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        return [float(x) * score for x in clean_scores(result, entity_candidates)], entity_candidates, entity_candidates_id
    elif args.prune_tools == "gnn":
        topn_entities, topn_scores = gnn_based_selection(question, entity_candidates_id, relation, args.width)
        
    elif args.prune_tools == "bm25":
        topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, args.width)
    else:
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id

    
def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def generate_final_answer_using_subquestion(question, context, args):
    context = '\n'.join(context)
    prompt = answer_using_subquestion_prompt + '\n- **Original Question:**' + question + '\n' + context + '\n **Final Output:**'
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return result


def half_stop(question, cluster_chain_of_entities, depth, args, context=[], sub_question=''):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    if depth == 1:
        answer = generate_answer(question, cluster_chain_of_entities, args)
    else:
        answer = generate_answer(sub_question, cluster_chain_of_entities, args)
        answer = '**Answer:** ' + answer
        context.append(answer)
        answer = generate_final_answer_using_subquestion(question, context, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset + '_' + args.LLM_type, context=context)


def generate_answer(question, cluster_chain_of_entities, args): 
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    return result


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))
    # id2entity_name_or_type
    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def wiki_retrieval(question, input_entity):
    pre_relations = []
    pre_heads= [-1] * len(input_entity)
    results = []
    retrieve_relations_with_scores = relation_search_prune(topic_entity[input_entity], input_entity, pre_relations, pre_heads[-1], question)
    for entity in retrieve_relations_with_scores:
        if entity['head']:
            entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
        else:
            entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)
        scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'])
        if len(entity_candidates) == 0:
            continue
        else:
            if entity['head']:
                for entity_id in entity_candidates_id:
                    results.append((entity['relation'], entity['entity'], id2entity_name_or_type(entity_id)))
            else:
                for entity_id in entity_candidates_id:
                    results.append((id2entity_name_or_type(entity_id), entity['relation'], entity['entity']))
    return results


def wiki_retrieval_with_backtrack(question, input_entity, args):
    """use backtrack module reasoning_with_backtrack to do retrieval"""
    path, status = reasoning_with_backtrack(
        question=question,
        start_entity_id=input_entity,
        # start_entity_name=input_entity,
        llm=args.llm_client,
        max_depth=args.max_depth,
        max_steps=args.max_steps,
        k_candidates=args.k_candidates,
    )
    results = []
    for node in path:
        for candidate in node.candidates:
            results.append((candidate['relation'], node.entity_id, id2entity_name_or_type(candidate['entity'])))
    return results

def reasoning(question, cluster_chain_of_entities, args, context=[], sub_question=''):
    if sub_question:
        prompt = prompt_evaluate + sub_question
    else:
        prompt = prompt_evaluate + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '

    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type)
    
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response
    


