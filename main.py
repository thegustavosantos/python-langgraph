import re
import os
from google import genai

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict

from numpy import rint

PROMPT_REACT = """
Você é um assistente de consulta de estoque e preços.
Consulte as ferramentas antes de informar quantidades ou preços.
Nunca invente valores ou Observações: elas serão enviadas pelo programa.

Ferramentas disponíveis:
- consultar_estoque[produto]: consulta a quantidade disponível.
- consultar_preco_produto[produto]: consulta o preço unitário.

Use exatamente estes nomes de produtos, no singular:
monitor, teclado, mouse gamer, webcam, headset, impressora.
Normalize plurais e sinônimos: "mouses gamers" corresponde a "mouse gamer".
Para um produto desconhecido, consulte a ferramenta e informe o resultado recebido.

Em cada turno, responda em apenas um dos formatos abaixo, sem Markdown,
sem explicações adicionais e sem expor raciocínio interno:

1. Se precisar consultar uma ferramenta, emita uma única ação:
Ação: consultar_estoque[monitor]
PAUSA
Encerre o turno imediatamente após PAUSA e aguarde a Observação do programa.
Use o mesmo formato para consultar_preco_produto[produto].

2. Quando as Observações recebidas forem suficientes, responda em uma linha:
Resposta: <resposta à pergunta do usuário>

Se precisar de estoque e preço, consulte uma ferramenta por turno e aguarde
as duas Observações antes de responder. Os valores devem vir exclusivamente
das Observações desta conversa.

Exemplo de conversa (cada bloco é um turno separado):
Usuário: Quantos mouses gamers temos em estoque?
Assistente:
Ação: consultar_estoque[mouse gamer]
PAUSA
Programa: Observação: Temos 80 mouse gamers em estoque.
Assistente:
Resposta: Há 80 mouses gamers em estoque.
""".strip()

LLM_MODEL_NAME = "gemini-3-flash-preview"
load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

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
        # Constrói o prompt com todo o histórico de mensagens
        prompt = ""
        for msg in self.messages:
            prompt += f"{msg['role']}: {msg['content']}\n"
        
        # Envia para o Gemini
        response = client.models.generate_content(model=LLM_MODEL_NAME, contents=prompt)

        return response.text

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
    precos = {
        "monitor": 999.90,
        "teclado": 150.00,
        "mouse gamer": 99.50,
        "webcam": 120.00,
        "headset": 180.00,
        "impressora": 750.00
    }

    if produto in precos:
        return f"O preço de um(a) {produto} é R$ {precos[produto]:.2f}."
    else:
        return f"Produto '{produto}' não encontrado na lista de preços."

def run_react_agent(question: str, max_iterations: int = 5) -> str:

    chat = client.chats.create(
        model=LLM_MODEL_NAME,
        config={"system_instruction": PROMPT_REACT},
        history=[],
    )

    current_prompt = question

    for i in range(max_iterations):
        response = chat.send_message(current_prompt)
        response_text = response.text.strip()

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
            else:
                observacao = f"Erro: Ação '{action_name}' desconhecida."
            
            current_prompt = f"Observação: {observacao}"

            print(f"Executou ação: {action_name}({action_arg})")
            print(f"Observação: {observacao}\n")

        else:
            return f"Erro: O agente não conseguiu extrair uma Ação ou Resposta final após {i+1} iterações. Última resposta: {response_text}"
    
    return "Erro: Limite máximo de iterações atingido sem uma resposta final."

if __name__ == "__main__":
    question_1 = "Quantos mouses gamers estão no inventário?"
    print(f"***Interação 1: {question_1}***")
    resposta_1 = run_react_agent(question_1)
    print(f"\n**RESPOSTA FINAL DO AGENTE 1:** {resposta_1}\n")
    print("\n"+"="*50 + "\n")

