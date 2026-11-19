# Retorno Inteligente: previsão de retorno à oficina para troca de óleo

Projeto prático do Nanodegree de IA & ML da Atitus. Solução de Machine Learning que aprende o ritmo de uso de cada veículo a partir do histórico de visitas a oficinas e prevê **em quantos dias ele volta para a próxima troca de óleo**, comparando com a regra fixa do fabricante e entregando à oficina a lista de quem contatar na semana.

Especificação: `specs.md`. Tarefas: `TODO.md`. Relatório: `relatorio/relatorio.md`.

## Dados

Quatro fontes reais, pseudonimizadas (código no lugar de placa e cliente): oficina Cirocar (NF-e, 2020 a 2026), Oficina Mecânica Ampese, oficina vizinha (Lica / Matheus Ampese) e a frota de máquinas da Prefeitura de São Domingos do Sul (em horas de horímetro). Nenhum dado gerado entra no treino.

## Como executar

```bash
pip install -r requirements.txt
python src/preparar_dados.py   # lê data/brutos/*.xlsx, gera data/visitas.csv e data/intervalos.csv
python src/treinar.py          # réguas, modelos, Optuna, prova final, gráficos, models/
pytest                         # 13 testes
streamlit run app/app.py       # painel da oficina
```

## Estrutura

```
executar_tudo.py roda preparar, treinar e testes em um comando
data/brutos/     planilhas das oficinas (aba Visitas)
data/            visitas.csv e intervalos.csv gerados
src/             preparar_dados.py, modelos.py, treinar.py
models/          modelo_retorno.joblib (regra escolhida + melhor ML) e metricas.json
app/             app.py (Streamlit)
tests/           pytest
relatorio/       relatorio.md, gráficos e padroes_do_curso.md
.github/         testes automáticos a cada push (GitHub Actions)
CONTRIBUTING.md  como o grupo contribui
```

## Resultado em uma frase

A regra cega do fabricante ("todo carro volta em X meses") erra 190 dias na validação e 127 no teste. Dividir o km recomendado pelo km por dia de cada carro (a ideia do grupo) derruba o erro para 53 e 31 dias; o melhor modelo de ML (Ridge sobre as réguas, ajustado por Optuna) empata: 57 e 30 dias. O ganho grande vem da personalização pelo ritmo de uso, e isso é dito no relatório.

## Integrantes

Arthur Risson, Vitor Pires, Nathalia de Souza, Gabriela Ampese, Leonardo Cambrussi, Isadora Aguirre.
