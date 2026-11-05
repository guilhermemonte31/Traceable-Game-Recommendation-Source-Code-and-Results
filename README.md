# Traceable Digital Game Recommendation Using Knowledge Graphs, Semantic Query Expansion, and LLM-Based Cypher Generation

The repository contains the following templates:

- versão-padrão.py: A recommendation model developed using translation flow, intent extraction, term expansion, Cypher query generation, and querying the Neo4j graph;

- b1-busca-literal: consists of a literal word search model, which will use as its database a text file containing the same information structured in the knowledge graph. This model will clean the received message (removing articles, pronouns,
conjunctions, etc.) and, based on the filtered words, search for a literal match between those words and those present in the text file, returning the match with the highest score;

- b2-vectorbased-semantic-retreval.py: consists of a vector-based text retrieval model, which uses the same text file to transform the information using the Sentence Transformers all-MiniLM-L6-v2 model. The embeddings are normalized using FAISS,retrieving the game that achieves the highest cosine similarity  with respect to the submitted query;

- b3-kg-sem-expansao.py: follows a pipeline similar to the one proposed in this article, with the difference that it does not expand query terms, in order to analyze the impact of this strategy on the results of the main model. Thus, this third model extracts the main terms from the user’s question and generates a Cypher query that will be executed on the same knowledge graph, returning the games with the highest scores;

The results from each model are saved in differents json arquives with their identification