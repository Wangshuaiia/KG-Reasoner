import openai
import re


judge_prompt = """You are helping to build an encyclopedia database. Your task is to determine whether a given input string can be considered an entity on its own. Valid entities include proper names such as people, locations, organizations, or creative works (e.g., books, films, songs, etc.).

If the input is a valid entity by itself, output:
Yes

If the input is not a valid entity, output:
No
and then extract and return the substring(s) that are valid entities within the input. Output these as:
Entity: <entity string>

Follow this output format exactly. See the examples below:

Examples
Input: All Around The World
Output: Yes

Input: President of Vietnam
Output: Yes

Input: United States Congress
Output: Yes

Input: Forrest Gump: Original Motion Picture soundtrack
Output: No
Entity: Forrest Gump: Original Motion Picture

Input: Haley Joel Osment age
Output: No
Entity: Haley Joel Osment

Now, analyze the following input:

Input: {{your_input_here}}"""

client = openai.OpenAI(
    api_key="sk-YiJGBvnpwcwMh4NoKj3vRQ", base_url="http://coreml-llm-proxy.k8-prod2.ess.volvo.net/", default_headers={"trace":'{"name":"gonogo_application"}'} 
)
model = "azure-gpt-4o"
max_tokens = 4096

def judge_entity(query):
    prompt = judge_prompt.replace("{{your_input_here}}", query)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content
    if "Yes" in text:
        return query
    else:
        entity = re.search(r'Entity:\s*(.*)', text)
        if entity:
            return entity.group(1).strip()
        else:
            return ""

if __name__ == "__main__":
    # 测试
    test_queries = [
        "Angelina Jolie's dog",
        "President of USA",
        "United States",
        "Forrest Gump: Original Motion Picture soundtrack",
        "Haley Joel Osment's home"
    ]
    
    for query in test_queries:
        print(f"Query: {query} -> Entity: {judge_entity(query)}")
    
