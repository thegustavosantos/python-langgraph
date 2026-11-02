import re
import os
from openai import OpenAI

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict


PROMPT_REACT = """
Você é um agente ReAct que funciona em um ciclo de Pensamento, Ação, Pausa e Observação.
Ao final do ciclo, você fornece uma Resposta.
Use "Pensamento" apenas para uma justificativa breve da próxima ação, sem detalhar raciocínio interno.
Use "Ação" para executar ferramentas - e então retorne "PAUSA".
A "Observação" será o resultado da ação executada.
Nunca invente Observações, preços ou quantidades. Aguarde o programa executar a ferramenta.
Emita apenas uma Ação por turno e encerre imediatamente após PAUSA.
Use sempre Ação: nome_da_ferramenta[argumento], inclusive [] quando não houver argumento.
Não use dois-pontos entre o nome da ferramenta e o argumento, nem blocos Markdown.
Use os produtos no singular: monitor, teclado, mouse gamer, webcam, headset, impressora.
Normalize "mouses gamers" para "mouse gamer". Para outros produtos, consulte e informe o resultado.
Quando tiver todas as Observações necessárias, comece a resposta diretamente com "Resposta:".
Você pode consultar outras ferramentas em turnos seguintes antes da resposta final.
Nos exemplos abaixo, Observação é enviada pelo programa em um turno separado.

Ações disponíveis:
- consultar_estoque[teclado]: retorna a quantidade disponível de um item.
- consultar_preco_produto[mouse gamer]: retorna o preço unitário de um produto.
- encontrar_produto_mais_caro[]: retorna o nome e o maior preço unitário do catálogo; não requer argumentos.
- calcular_valor_total_lista[teclado, mouse gamer]: soma o preço de uma unidade de cada item, separado por vírgula.

Exemplo:
Pergunta: Quantos monitores temos em estoque?
Pensamento: Devo consultar a ação consultar_estoque para saber a quantidade de monitores.
Ação: consultar_estoque[monitor]
PAUSA

Observação: Temos 75 monitores em estoque.
Resposta: Há 75 monitores em estoque.

Exemplo:
Pergunta: Qual é o produto mais caro?
Pensamento: Preciso usar a ação encontrar_produto_mais_caro para descobrir qual produto tem o maior preço.
Ação: encontrar_produto_mais_caro[]
PAUSA

Observação: O produto mais caro é o(a) monitor com preço de R$ 999.90.
Resposta: O produto mais caro é o(a) monitor com preço de R$ 999.90.

Exemplo:
Pergunta: Quanto custa um teclado e um mouse gamer?
Pensamento: O usuário quer saber o valor total de vários itens. Devo usar a ação calcular_valor_total_lista com os itens "teclado, mouse gamer".
Ação: calcular_valor_total_lista[teclado, mouse gamer]
PAUSA

Observação: O valor total dos itens encontrados é R$ 249.50.
Resposta: O valor total do teclado e do mouse gamer é R$ 249.50.

""".strip()

load_dotenv()
LLM_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-6-luna")
PRECOS_DO_INVENTARIO = {
    "monitor": 999.90,
    "teclado": 150.00,
    "mouse gamer": 99.50,
    "webcam": 120.00,
    "headset": 180.00,
    "impressora": 750.00,
}
client = OpenAI()

class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        response = client.responses.create(
            model=LLM_MODEL_NAME,
            input=self.messages,
        )
        return response.output_text

class AgentState(TypedDict):
    question: str
    history: list[str]
    pending_action: str
    final_answer: str

def consultar_estoque(item: str) -> str:
    """
    Simula a consulta de estoque de um item no inventário.
    """
    item = item.lower()
    estoque = {
        "monitor": 75,
        "teclado": 120,
        "mouse gamer": 80,
        "webcam": 40,
        "headset": 60,
        "impressora": 15
    }

    if item in estoque:
        return f"Temos {estoque[item]} {item}s em estoque." #criamos uma máscara para a resposta, para ficar mais elegante e profissional
    else:
        return f"Item '{item}' não encontrado no inventário." #criamos também uma resposta para produtos não localizados no inventário

def consultar_preco_produto(produto: str) -> str:
    """
    Simula a consulta do preço unitário de um produto.
    """
    produto = produto.lower()
    precos = PRECOS_DO_INVENTARIO

    if produto in precos:
        return f"O preço de um(a) {produto} é R$ {precos[produto]:.2f}."
    else:
        return f"Produto '{produto}' não encontrado na lista de preços."

def encontrar_produto_mais_caro() -> str:
    """Retorna o produto de maior preço unitário; em empate, o primeiro do catálogo."""
    if not PRECOS_DO_INVENTARIO:
        return "Não há produtos na lista de preços."

    produto = max(PRECOS_DO_INVENTARIO, key=PRECOS_DO_INVENTARIO.get)
    preco = PRECOS_DO_INVENTARIO[produto]
    return f"O produto mais caro é o(a) {produto} com preço de R$ {preco:.2f}."


def calcular_valor_total_lista(lista_itens: str) -> str:
    """
    Calcula o valor total de uma lista de itens de compra.
    Recebe uma string com itens separados por virgula (ex: "teclado, mouse gamer, monitor").
    """
    
    precos_do_inventario = PRECOS_DO_INVENTARIO

    itens_processados = [item.strip().lower() for item in lista_itens.split(',')]
    
    valor_total = 0.0
    itens_nao_encontrados = []

    for item in itens_processados:
        if item in precos_do_inventario:
            valor_total += precos_do_inventario[item]
        else:
            itens_nao_encontrados.append(item)

    resposta = f"O valor total dos itens encontrados é R$ {valor_total:.2f}."
    if itens_nao_encontrados:
        resposta += f" Os seguintes itens não foram encontrados e não foram incluídos no cálculo: {', '.join(itens_nao_encontrados)}"
        
    return resposta


def run_react_agent(question: str, max_iterations: int = 5) -> str:

    messages = []

    current_prompt = question

    for i in range(max_iterations):
        messages.append({"role": "user", "content": current_prompt})
        response = client.responses.create(
            model=LLM_MODEL_NAME,
            instructions=PROMPT_REACT,
            input=messages,
        )
        response_text = response.output_text.strip()
        if not response_text:
            return "Erro: O modelo não retornou uma resposta de texto."
        messages.append({"role": "assistant", "content": response_text})

        print(f"--- Iteração {i+1} ---")
        print(f"Modelo pensou/respondeu:\n{response_text}\n")

        if response_text.startswith("Resposta:"):
            return response_text.replace("Resposta:", "").strip()

        match = re.search(r"Ação:\s*(\w+)\s*\[\s*(.*)\s*\]", response_text)

        if match:
            action_name = match.group(1).strip()
            action_arg = match.group(2).strip()

            observacao = ""
            if action_name == "consultar_estoque":
                observacao = consultar_estoque(action_arg)
            elif action_name == "consultar_preco_produto":
                observacao = consultar_preco_produto(action_arg)
            elif action_name == "encontrar_produto_mais_caro":
                observacao = encontrar_produto_mais_caro()
            elif action_name == "calcular_valor_total_lista": # Adição da nova ferramenta
                observacao = calcular_valor_total_lista(action_arg)
            else:
                observacao = f"Erro: Ação '{action_name}' desconhecida."
            
            current_prompt = f"Observação: {observacao}"

            print(f"Executou ação: {action_name}({action_arg})")
            print(f"Observação: {observacao}\n")

        else:
            return f"Erro: O agente não conseguiu extrair uma Ação ou Resposta final após {i+1} iterações. Última resposta: {response_text}"
    
    return "Erro: Limite máximo de iterações atingido sem uma resposta final."

def iniciar_conversacao_com_agente():
    print("--- Agente de Inventário Interativo ---")
    print("Digite sua pergunta sobre o inventário, ou digite 'sair' para encerrar.")
    print("*" * 50)

    while True:
        pergunta_usuario = input("\nVocê: ")

        if pergunta_usuario.lower().strip() == 'sair':
            print("Encerrando a conversa. Até logo!")
            break
        
        print("\nAgente: Processando...")
        try:
            # Chama a sua função run_react_agent com a pergunta do usuário
            resposta_agente = run_react_agent(pergunta_usuario)
            print(f"\nAgente: {resposta_agente}")
        except Exception as e:
            # Captura erros para evitar que o loop seja interrompido
            print(f"\nAgente: Ocorreu um erro ao processar sua pergunta: {e}")
            print("Por favor, tente novamente ou digite 'sair'.")

if __name__ == "__main__":
# Interação 1: Consultar Estoque
    pergunta_1 = "Quantos teclados temos em estoque?"
    print(f"\n**Interação 1: {pergunta_1}**")
    resposta_1 = run_react_agent(pergunta_1)
    print(f"\n**RESPOSTA FINAL DO AGENTE 1:** {resposta_1}\n")

    print("\n" + "="*80 + "\n")

    # Interação 2: Consultar Preço
    pergunta_2 = "Qual o preço de um headset?"
    print(f"\n**Interação 2: {pergunta_2}**")
    resposta_2 = run_react_agent(pergunta_2)
    print(f"\n**RESPOSTA FINAL DO AGENTE 2:** {resposta_2}\n")

    print("\n" + "="*80 + "\n")

    # Interação 3: Item não encontrado no estoque
    pergunta_3 = "Temos cadeiras em estoque?"
    print(f"\n**Interação 3: {pergunta_3}**")
    resposta_3 = run_react_agent(pergunta_3)
    print(f"\n**RESPOSTA FINAL DO AGENTE 3:** {resposta_3}\n")

    print("\n" + "="*80 + "\n")
    # Interação 4: Encontrar o Produto Mais Caro
    pergunta_4 = "Qual é o produto mais caro?"
    print(f"\n**Interação 4: {pergunta_4}**")
    resposta_4 = run_react_agent(pergunta_4)
    print(f"\n**RESPOSTA FINAL DO AGENTE 4:** {resposta_4}\n")

    print("\n" + "="*80 + "\n")
    #Interação 5: Calcular Valor Total da Lista (NOVA FUNCIONALIDADE)
    pergunta_5 = "Qual o valor de um teclado, uma impressora e uma webcam?"
    print(f"\n**Interação 5: {pergunta_5}**")
    resposta_5 = run_react_agent(pergunta_5)
    print(f"\n**RESPOSTA FINAL DO AGENTE 5:** {resposta_5}\n")

    print("\n--- Fim das Interações ---")

    iniciar_conversacao_com_agente()
