# RAG FII

## Descrição

Projeto para estudar o ganho de desempenho da arquitetura RAG em comparação com os LLMs convencionais no domínio de fundos imobiliários

## Requisitos
- Python 3.8 ou superior instalado em seu ambiente.
- Uma chave de API da OpenAI (pode ser obtida aqui).
- pip para instalação de dependências.


## Passo a passo para rodar localmente

### 1. Clone o repositório

```git clone https://github.com/caiodeandrade/rag-fii.git```


```cd rag-fii```

### 2. Crie e ative um ambiente virtual

Linux/macOS:

```python3 -m venv .venv```

```source .venv/bin/activate```

Windows:

```python -m venv .venv```

```.venv\Scripts\activate```

### 3. Instale as dependências

```pip install -r requirements.txt```

### 4. Adicione sua chave da OpenAI no arquivo .env

Crie um arquivo chamado .env na raiz do projeto com o seguinte conteúdo:

```OPENAI_API_KEY=<sua-chave-da-openai-aqui>```

### 5. Adicione o PDF que deseja consultar

Coloque seu PDF na pasta assets/ e ajuste o caminho do arquivo na variável pdf_path do main.py, se necessário.

### 6. Rode a aplicação

```python main.py```

O servidor estará disponível em http://localhost:5000.


## Como usar

Envie uma requisição POST para o endpoint /query com um JSON contendo a chave question.
Exemplo usando curl:

```
curl -X POST http://localhost:5000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "Qual o tema do capítulo 2?"}'
```

A resposta será um JSON com a resposta gerada pelo modelo.

<hr>

Qualquer dúvida ou problema, fique à vontade para abrir uma issue!
