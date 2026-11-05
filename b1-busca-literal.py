
import os
import json
import re
import unicodedata
from collections import Counter



PASTA_PROJETO = os.path.dirname(
    os.path.abspath(__file__)
)

ARQUIVO_JOGOS = os.path.join(
    PASTA_PROJETO,
    "b2-text.txt"
)

ARQUIVO_RESULTADOS = os.path.join(
    PASTA_PROJETO,
    "teste-b1.json"
)


# NORMALIZAÇÃO DO TEXTO
def normalizar_texto(texto):
    """
    Converte o texto para minúsculas, remove acentos
    e extrai palavras.
    """

    texto = texto.lower()

    texto = unicodedata.normalize(
        "NFKD",
        texto
    )

    texto = "".join(
        caractere
        for caractere in texto
        if not unicodedata.combining(caractere)
    )

    palavras = re.findall(
        r"\b[a-z0-9]+\b",
        texto
    )

    return palavras


# LEITURA DO ARQUIVO DE JOGOS
def ler_arquivo_jogos(caminho):

    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
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

        for linha in bloco.splitlines():

            linha = linha.strip()

            match = re.match(
                r"^([A-Z_]+):\s*(.*)$",
                linha
            )

            if match:

                campo = match.group(1).lower()
                valor = match.group(2).strip()

                jogo[campo] = valor

        if "name" in jogo:

            jogo["texto_completo"] = bloco

            jogos.append(jogo)

    return jogos


# BUSCA LITERAL

def buscar_jogo_literal(questao, jogos):

    palavras_pergunta = normalizar_texto(
        questao
    )

    palavras_ignoradas = {
        "a", "an", "the", "and", "or",
        "i", "can", "where", "what",
        "looking", "for", "want", "with",
        "is", "in", "of", "to", "on",
        "game", "games", "that", "where", "can", "be", "are"
    }

    termos_busca = set(
        palavra
        for palavra in palavras_pergunta
        if palavra not in palavras_ignoradas
    )

    if not termos_busca:
        print("A pergunta não contém termos úteis para a busca.")
        return {
            "pergunta": questao,
            "jogo_encontrado": None,
            "palavras_correspondentes": 0
        }

    melhor_jogo = None
    melhor_pontuacao = 0
    resultados = []

    for jogo in jogos:

        texto_jogo = normalizar_texto(
            jogo["texto_completo"]
        )
        print("Teste:: ", texto_jogo)

        contagem = Counter(texto_jogo)

        termos_encontrados = {
            termo
            for termo in termos_busca
            if contagem[termo] > 0
        }

        pontuacao = len(termos_encontrados)

        resultados.append({
            "jogo": jogo,
            "pontuacao": pontuacao,
            "termos": sorted(termos_encontrados)
        })

        if pontuacao > melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor_jogo = jogo

    print("\n" + "=" * 60)
    print("RESULTADO DA BUSCA LITERAL")
    print("=" * 60)

    print(f"Pergunta: {questao}")
    print(f"Termos pesquisados: {sorted(termos_busca)}")

    if melhor_jogo is None:

        print("\nNenhum jogo contém os termos pesquisados.")
        return {
            "pergunta": questao,
            "jogo_encontrado": None,
            "palavras_correspondentes": 0
        }

    print(f"\nJogo encontrado: {melhor_jogo['name']}")
    print(f"Palavras correspondentes: {melhor_pontuacao}")
    print(
        "Termos encontrados:",
        next(
            resultado["termos"]
            for resultado in resultados
            if resultado["jogo"] is melhor_jogo
        )
    )

    return {
        "pergunta": questao,
        "jogo_encontrado": melhor_jogo["name"],
        "palavras_correspondentes": melhor_pontuacao
    }


def salvar_resultados(resultados, caminho):
    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(resultados, arquivo, ensure_ascii=False, indent=4)





def main():

    jogos = ler_arquivo_jogos(
        ARQUIVO_JOGOS
    )

    if not jogos:
        raise ValueError(
            "Nenhum jogo foi encontrado no arquivo."
        )

    print(f"Jogos carregados: {len(jogos)}")
    resultados_pesquisa = []

    while True:
        questao = input(
            "\nDigite sua pergunta (ou 'sair' para encerrar): "
        ).strip()

        if questao.lower() == "sair":
            salvar_resultados(resultados_pesquisa, ARQUIVO_RESULTADOS)
            print(f"Resultados salvos em: {ARQUIVO_RESULTADOS}")
            print("Encerrando a busca.")
            break

        if not questao:
            print("Digite uma pergunta para iniciar a busca.")
            continue

        resultado = buscar_jogo_literal(questao, jogos)
        resultados_pesquisa.append(resultado)


if __name__ == "__main__":
    main()
