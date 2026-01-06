# config.py

OLLAMA_URL = "http://your-service/v1/chat/completions"
OLLAMA_HEADERS = {"Content-Type": "application/json"}

LLAMA_31_70B_BASE_URL = "http://your-service/v1/"

COREML_PROXY_URL = "http://your-service"

DEFAULT_SYSTEM_PROMPT = "You are an AI assistant that helps people find information."

DEFAULT_OPENAI_API_KEY = "your_api_key"

import torch
MODEL_CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "bert_name": "bert-base-uncased",
    "bert_hidden_dim": 768,
    "gnn_hidden_dim": 128,
    "num_classes": 2,
    "model_state_path": "your_model_path.pt"
}

SPARQL_CONFIG = {
    "SPARQLPATH": "http://your-sparql-endpoint",
    "sparql_head_relations": """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation
WHERE {
  ns:%s ?relation ?x .
}""",
    "sparql_tail_relations": """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation
WHERE {
  ?x ?relation ns:%s .
}""",
    "sparql_tail_entities_extract": """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity
WHERE {
  ns:%s ns:%s ?tailEntity .
}""",
    "sparql_head_entities_extract": """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity
WHERE {
  ?tailEntity ns:%s ns:%s .
}""",
    "sparql_id": """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
  {
    ?entity ns:type.object.name ?tailEntity .
    FILTER(?entity = ns:%s)
  }
  UNION
  {
    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
    FILTER(?entity = ns:%s)
  }
}"""
}