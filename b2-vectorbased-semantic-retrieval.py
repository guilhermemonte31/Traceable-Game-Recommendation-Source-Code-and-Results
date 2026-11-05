import re
import json
import os

import faiss
from sentence_transformers import SentenceTransformer



PASTA_PROJETO = os.path.dirname(
    os.path.abspath(__file__)
)

ARQUIVO_JOGOS = os.path.join(
    PASTA_PROJETO,
    "b2-text.txt"
)


ARQUIVO_RESULTADOS = os.path.join(
    PASTA_PROJETO,
    "resultados-b2.json"
)

MODELO_EMBEDDING = "all-MiniLM-L6-v2"

DIMENSAO_EMBEDDING = 384


# CARREGA MODELO

print("Carregando modelo de embedding...")

model = SentenceTransformer(
    MODELO_EMBEDDING
)

print("Modelo carregado.")


# LEITURA DO ARQUIVO DE JOGOS

def ler_arquivo_jogos(caminho):

    print("\nLendo arquivo:")
    print(caminho)

    if not os.path.exists(caminho):

        raise FileNotFoundError(
            f"\nArquivo não encontrado:\n{caminho}"
        )

    with open(
        caminho,
        "r",
        encoding="utf-8"
    ) as arquivo:

        conteudo = arquivo.read()

    blocos = re.split(
        r"\n\s*---\s*\n",
        conteudo
    )

    jogos = []

    for bloco in blocos:

        bloco = bloco.strip()

        if not bloco:
            continue

        jogo = {}

        linhas = bloco.splitlines()

        campo_atual = None
        valor_campo = []

        for linha in linhas:

            linha = linha.strip()

            if not linha:
                continue

            match = re.match(
                r"^([A-Z_]+):\s*(.*)$",
                linha
            )

            if match:

                if campo_atual is not None:
                    jogo[campo_atual] = (
                        " ".join(valor_campo).strip()
                    )

                campo_atual = (
                    match.group(1).lower()
                )

                valor_campo = [
                    match.group(2).strip()
                ]

            else:

                if campo_atual is not None:

                    valor_campo.append(
                        linha
                    )

        if campo_atual is not None:
            jogo[campo_atual] = (
                " ".join(valor_campo).strip()
            )

        if "name" in jogo:

            jogos.append(jogo)

    print(
        f"{len(jogos)} jogos encontrados."
    )

    return jogos


# CRIA TEXTO PARA O EMBEDDING
def criar_texto_embedding(jogo):

    campos = [

        "name",
        "plot",
        "progression",
        "genre",
        "subgenre",
        "mechanics",
        "coreloop",
        "theme",
        "atmosphere",
        "difficulty",
        "setting",
        "perspective",
        "gamemode",
        "language"

    ]

    partes = []
    for campo in campos:

        valor = jogo.get(campo)

        if valor:
            partes.append(
                f"{campo}: {valor}"
            )

    return "\n".join(partes)


# GERA EMBEDDINGS DOS JOGOS

def gerar_embeddings(jogos):

    print("\nGerando embeddings dos jogos...")

    textos = []

    for jogo in jogos:

        texto = criar_texto_embedding(
            jogo
        )

        textos.append(texto)

    embeddings = model.encode(
        textos,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings.astype(
        "float32"
    )


# CRIA ÍNDICE FAISS

def criar_indice(embeddings):

    print("\nCriando índice FAISS...")
    indice = faiss.IndexFlatIP(
        DIMENSAO_EMBEDDING
    )

    indice.add(
        embeddings
    )

    print(
        f"{indice.ntotal} embeddings adicionados."
    )

    return indice


# GERA EMBEDDING DA QUESTÃO

def gerar_embedding_questao(questao):

    embedding = model.encode(
        questao,
        normalize_embeddings=True
    )

    embedding = embedding.astype(
        "float32"
    )

    return embedding.reshape(
        1,
        DIMENSAO_EMBEDDING
    )


# BUSCA O JOGO MAIS SIMILAR

def buscar_jogo(questao, indice, jogos):

    print("\n")
    print("=" * 60)
    print("CONSULTA")
    print("=" * 60)

    print(
        f"Questão: {questao}"
    )


    embedding_questao = (
        gerar_embedding_questao(
            questao
        )
    )
    scores, indices = indice.search(
        embedding_questao,
        1
    )

    indice_jogo = indices[0][0]

    score = scores[0][0]


    jogo = jogos[indice_jogo]


    print("\n")
    print("=" * 60)
    print("JOGO MAIS SIMILAR")
    print("=" * 60)

    print(
        f"Jogo: {jogo['name']}"
    )

    print(
        f"Similaridade: {score:.4f}"
    )

    return {
        "pergunta": questao,
        "jogo_encontrado": jogo["name"],
        "similaridade": float(score)
    }


def salvar_resultados(resultados, caminho):
    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(resultados, arquivo, ensure_ascii=False, indent=4)



def main():

    # 1. Lê os jogos
    jogos = ler_arquivo_jogos(
        ARQUIVO_JOGOS
    )

    if not jogos:

        raise ValueError(
            "Nenhum jogo foi encontrado."
        )

    # 2. Gera embeddings dos jogos
    embeddings = gerar_embeddings(
        jogos
    )

    # 3. Cria índice FAISS
    indice = criar_indice(
        embeddings
    )

    resultados_pesquisa = []

    while True:
        questao = input(
            "\nDigite sua pergunta (ou 'sair' para encerrar): "
        ).strip()

        if questao.lower() == "sair":
            salvar_resultados(resultados_pesquisa, ARQUIVO_RESULTADOS)
            print(f"Resultados salvos em: {ARQUIVO_RESULTADOS}")
            break

        if not questao:
            print("Digite uma pergunta para iniciar a busca.")
            continue

        resultados_pesquisa.append(
            buscar_jogo(questao, indice, jogos)
        )
        
if __name__ == "__main__":

    main()