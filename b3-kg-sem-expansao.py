from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0,
    openai_api_key=API_KEY,
    openai_api_base="https://api.groq.com/openai/v1"
)

graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

ARQUIVO_RESULTADOS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "resultados-b3.json"
)

CYPHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a Neo4j expert.

Generate ONLY a valid Cypher query.
Do not include explanations, markdown, comments, or extra text.

==================================================
DATABASE STRUCTURE
==================================================

DIRECT PROPERTIES OF :Game

Use direct properties only for name, plot and progression.

Examples: g.name, g.plot, g.progression

Do not use properties like: g.genre, g.theme, g.mechanics, g.setting, g.perspective, g.language


These are stored as related nodes.

==================================================
RELATED NODES (USE RELATIONSHIPS)
==================================================

Use relationships for: Genre, Subgenre, Mechanic, GameMode, Theme, Atmosphere, CoreLoop, Difficulty, Setting, Perspective, Language

Examples:

(g:Game)-[:HAS_GENRE]->(ge:Genre)
(g:Game)-[:HAS_SUBGENRE]->(sg:Subgenre)
(g:Game)-[:HAS_MECHANIC]->(m:Mechanic)
(g:Game)-[:HAS_GAMEMODE]->(gm:GameMode)
(g:Game)-[:HAS_THEME]->(t:Theme)
(g:Game)-[:HAS_ATMOSPHERE]->(a:Atmosphere)
(g:Game)-[:HAS_CORELOOP]->(c:CoreLoop)
(g:Game)-[:HAS_DIFFICULTY]->(d:Difficulty)
(g:Game)-[:HAS_SETTING]->(s:Setting)
(g:Game)-[:HAS_PERSPECTIVE]->(p:Perspective)
(g:Game)-[:HAS_LANGUAGE]->(l:Language)

     
==================================================
TERM EXPANSION INPUT
==================================================

Besides the user question, you will also receive a dictionary called:

expanded_terms

This dictionary contains preprocessed semantic terms already classified
by attribute.

Example:

{{
  "mechanics_terms": ["horse riding", "horseback riding"],
  "genre_terms": ["action", "adventure"],
  "setting_terms": ["open world", "rural"],
  "coreloop_terms": ["mission based"],
  "theme_terms": ["exploration"],
  "atmosphere_terms": ["realistic"],
  "perspective_terms": ["third person"]
}}
you must use these terms in the MATCH to know which nodes to match and in the WHERE clause to filter them.
     
==================================================
PARAMETER USAGE (MANDATORY)
==================================================

Never use fixed literal values inside WHERE clauses.

Wrong:

WHERE toLower(t.name) CONTAINS "morality"

Correct:

WHERE ANY(term IN $theme_terms
WHERE toLower(t.name) CONTAINS toLower(term))

Always use parameter placeholders like:

$genre_terms
$theme_terms
$mechanics_terms
$setting_terms
$perspective_terms
$subgenre_terms
$difficulty_terms
$coreloop_terms
$atmosphere_terms
$gamemode_terms
$language_terms

For direct properties like plot and progression, use:

ANY(term IN $plot_terms
WHERE toLower(g.plot) CONTAINS toLower(term))

==================================================
SEMANTIC MATCHING
==================================================

Support synonyms and wording variations using:

toLower(node.name) CONTAINS toLower(term)

Example:

"urban setting" → city, metropolitan, urban

Use semantic flexibility through params, never fixed literals.

==================================================
EXPECTED QUERY STYLE
==================================================

Generate ranking-based recommendation queries.

Do NOT use rigid filtering with:

WHERE condition1 AND condition2 AND condition3

Instead, use score-based ranking.

Use:

- OPTIONAL MATCH
- CASE WHEN
- score aggregation with SUM
- ORDER BY total_score DESC

The goal is:

Return the rank of games based on their ranking score.

Example:

MATCH (g:Game)

OPTIONAL MATCH (g)-[:HAS_MECHANIC]->(m:Mechanic)


WITH g,
SUM(
  CASE
  WHEN ANY(term IN ['horse riding', 'horseback riding', 'mount riding', 'equestrian', 'exploration', 'crafting']
  WHERE toLower(m.name) CONTAINS toLower(term))
  THEN 1
  ELSE 0
  END
) AS mechanics_score

OPTIONAL MATCH (g)-[:HAS_CORELOOP]->(c:CoreLoop)
WITH g, mechanics_score,
SUM(
  CASE
  WHEN ANY(term IN ['rewards', 'fight', 'combat', 'missions', 'exploration', 'crafting']
  WHERE toLower(c.name) CONTAINS toLower(term))
  THEN 1
  ELSE 0
  END
) AS coreloop_score

OPTIONAL MATCH (g)-[:HAS_GENRE]->(ge:Genre)

WITH g,mechanics_score, coreloop_score,
SUM(
  CASE
  WHEN ANY(term IN ['action', 'adventure']
  WHERE toLower(ge.name) CONTAINS toLower(term))
  THEN 1
  ELSE 0
  END
) AS genre_score


WITH g,
mechanics_score + genre_score + coreloop_score AS total_score

WHERE total_score > 0

RETURN
g.name,
total_score

ORDER BY total_score DESC
LIMIT 10


==================================================
FINAL RULE
==================================================

If you generate:

WHERE something = "value"

or

WHERE CONTAINS "value"

the query is wrong.

Always prefer recommendation ranking over strict filtering.

Use:

CASE
WHEN ANY(term IN ...)
THEN 1
ELSE 0
END

instead of:

WHERE A AND B AND C

Return best matches ordered by score.

for related nodes.
"""),

    ("user", """
    Question:
    {question}

    Expanded Terms:
    {expanded_terms}

    Cypher:
    """)
])

def limpar_resposta_llm(texto):
    return (
        texto.replace("```python", "")
             .replace("```json", "")
             .replace("```", "")
             .strip()
    )


def extrair_termos(texto):
   
    prompt = f"""
You are a system that extracts search terms for a game database.

TASK:
- Extract key concepts from the sentence
- Classify each concept by attribute
- Use ONLY the words/concepts that are present in the sentence

RULES:
- Do NOT expand terms
- Do NOT add synonyms
- Do NOT add variations or related terms
- Return ONLY a valid JSON
- Do NOT use markdown
- Do NOT use ```python
- Do NOT use ```json

Keys = property names (mechanics, genre, setting, subgenre, coreloop, 
    theme, atmosphere, perspective, gamemode, language, difficulty)
Values = list of terms

Example:

Sentence: games with vehicle driving in an urban setting

{{
  "mechanics": ["vehicle driving"],
  "setting": ["urban"]
}}

Now process:
{texto}
"""

    resposta = llm.invoke(prompt)


    conteudo_limpo = limpar_resposta_llm(resposta.content)


    try:
        return json.loads(conteudo_limpo)
    except Exception as e:
        print("Erro ao converter JSON:", e)
        return {}


def traduzir_pt_en(texto):
    prompt = f"""
You are a translator specialized in systems that query a Neo4j graph database about video games.

Your task is to translate the user query from Portuguese to English.

IMPORTANT RULES:
- Return ONLY the translated sentence.
- Do NOT explain anything.
- Do NOT add extra text.
- Preserve the original intent exactly.
- Use simple and clear English.
- Adapt terms to match a game database context.

STANDARDIZE TERMS:
- "jogos" → "games"
- "gênero" → "genre"
- "subgênero" → "subgenre"
- "mecânicas" → "mechanics"
- "modo de jogo" → "game mode"
- "tema" → "theme"
- "atmosfera" → "atmosphere"
- "core loop" → "core loop"
- "dificuldade" → "difficulty"
- "enredo" → "plot"
- "progressão" → "progression"
- "ambientação" → "setting"
- "perspectiva" → "perspective"
- "idioma" → "language"

EXAMPLES:
Portuguese: liste todos os jogos e seus gêneros
English: list all games and genres they have

Portuguese: liste jogos de ação com multiplayer
English: list games of action genre with multiplayer 

Portuguese: jogos com mecânicas de exploração
English: games with exploration mechanics

Generate the English translation in all lowercase letters

Now translate:

{texto}
"""
    
    resposta = llm.invoke(prompt)
    print("Tradução:", resposta.content)
    return resposta.content.strip()



def montar_parametros(termos):
    params = {}
    for chave, valores in termos.items():
        params[f"{chave}_terms"] = valores
    return params


def executar_busca(pergunta_pt, chain):
    pergunta_en = traduzir_pt_en(pergunta_pt)
    print("\n\nPergunta traduzida:", pergunta_en)


    termos = extrair_termos(pergunta_en)
    print("\n\nTermos extraídos (sem expansão):", termos)

    params = montar_parametros(termos)
    print("\n\nParams:", params)
    print("\n\nExecutando consulta...\n\n")

    cypher = chain.cypher_generation_chain.invoke({
        "question": pergunta_en,
        "expanded_terms": params,
        **params
    })

    print("Cypher gerado:")
    print(cypher)
    print("\n\nExecutando consulta no Neo4j...\n\n")

    resultado = graph.query(cypher, params=params)
    print("\n\nResultado final:", resultado)

    top10 = []
    for item in resultado[:10]:
        nome_jogo = item.get("game")
        pontuacao = item.get("total_score")
        top10.append({
            "jogo": nome_jogo,
            "pontuacao": pontuacao
        })

    return {
        "pergunta": pergunta_pt,
        "top10": top10
    }


def salvar_resultados(resultados):
    with open(ARQUIVO_RESULTADOS, "w", encoding="utf-8") as arquivo:
        json.dump(resultados, arquivo, ensure_ascii=False, indent=4)


def main():
    print("Iniciando busca de jogos.\n")

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=CYPHER_PROMPT,
        allow_dangerous_requests=True
    )

    resultados_pesquisa = []

    while True:
        pergunta_pt = input(
            "\nDigite sua pergunta (ou 'sair' para encerrar): "
        ).strip()

        if pergunta_pt.lower() == "sair":
            salvar_resultados(resultados_pesquisa)
            print(f"Resultados salvos em: {ARQUIVO_RESULTADOS}")
            break

        if not pergunta_pt:
            print("Digite uma pergunta para iniciar a busca.")
            continue

        try:
            resultados_pesquisa.append(
                executar_busca(pergunta_pt, chain)
            )
        except Exception as erro:
            print(f"Erro ao processar a pergunta: {erro}")


if __name__ == "__main__":
    main()