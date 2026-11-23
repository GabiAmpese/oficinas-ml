# Como contribuir (guia para o grupo)

Este projeto é de todos. Qualquer um pode mudar código, dados, relatório ou app. A regra é uma só: **tudo passa pelo Git**, para que ninguém perca o trabalho do outro e o histórico mostre quem fez o quê.

## 1. Primeira vez: baixar o projeto

Instale o [Git](https://git-scm.com/downloads) e o Python 3.11 ou mais novo. No terminal (no VS Code: Terminal > Novo Terminal):

```bash
git clone https://github.com/<usuario>/oficinas-ml.git
cd oficinas-ml
pip install -r requirements.txt
python executar_tudo.py --app
```

Se o painel abrir no navegador, está tudo funcionando.

## 2. Fazer uma mudança (o ciclo de sempre)

```bash
git pull                                  # 1. pega o que os outros já fizeram
git checkout -b minha-mudanca             # 2. cria um "ramo" só seu (troque o nome)
# ... edite os arquivos ...
python executar_tudo.py                   # 3. confira que tudo continua rodando e os testes passam
git add .                                 # 4. marca o que mudou
git commit -m "Explica em uma frase o que você mudou"   # 5. salva com mensagem
git push -u origin minha-mudanca          # 6. envia para o GitHub
```

Depois, no site do GitHub, abra um **Pull Request** do seu ramo para `main`. Um colega lê, comenta se quiser, e aprova. O GitHub roda os testes sozinho em cada Pull Request (pasta `.github/workflows`); se ficar vermelho, algo quebrou.

Mudança pequena e segura (um texto no relatório, por exemplo) pode ir direto na `main`:

```bash
git pull
# edite
git add . && git commit -m "Corrige texto da seção 5 do relatório" && git push
```

## 3. Onde mexer para cada coisa

| Quero... | Arquivo |
|---|---|
| Acrescentar visitas de uma oficina | a planilha dela em `data/brutos/`, aba Visitas (depois rode `python executar_tudo.py`) |
| Adicionar uma oficina nova | nova planilha `data/brutos/NOME.xlsx` no mesmo formato; se for em horas, inclua o nome em `FONTES_EM_HORAS` em `src/preparar_dados.py` |
| Mudar o limite de "troca perdida" ou os intervalos padrão | constantes no topo de `src/preparar_dados.py` |
| Mudar os custos da decisão (R$ do serviço perdido e do contato) | `CUSTO_TARDE` e `CUSTO_CEDO` em `src/treinar.py` |
| Testar outro modelo ou outras features | `src/modelos.py` (`FEATURES`, `construir_modelo`) e a lista `candidatos` em `src/treinar.py` |
| Mudar o número de blocos da validação temporal | `dobras_temporais` em `src/treinar.py` |
| Mudar o painel | `app/app.py` |
| Mudar o relatório | `relatorio/relatorio.md` |

## 4. Regras do projeto

- **Só dados reais.** Nenhuma linha inventada ou gerada entra nas planilhas.
- **Sem dado pessoal.** Nunca placa completa, nome, CPF ou telefone. Só o código do veículo (V001...). A tabela "código x cliente" fica na oficina, fora do repositório.
- **Testes passando.** `pytest` precisa ficar verde antes de enviar. Se mudou uma regra de negócio, mude ou acrescente o teste correspondente em `tests/`.
- **Mensagem de commit em português, no presente, dizendo o quê e por quê.** Exemplo: "Aumenta limite de troca perdida para 18.000 km porque a Cirocar informou tolerância maior".
- **Não comite** `__pycache__`, ambientes virtuais nem arquivos do seu editor (já estão no `.gitignore`).

## 5. Deu conflito?

Se o `git pull` reclamar de conflito, abra o arquivo: o Git marca os dois trechos com `<<<<<<<` e `>>>>>>>`. Escolha o que fica, apague as marcas, salve, `git add .` e `git commit`. Em dúvida, pergunte no grupo antes de forçar qualquer coisa.
