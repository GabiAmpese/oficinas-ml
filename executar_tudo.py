"""
executar_tudo.py — Roda o projeto inteiro em um comando:
preparar os dados, treinar, testar e (opcional) abrir o app.

    python executar_tudo.py          # prepara, treina e testa
    python executar_tudo.py --app    # faz tudo isso e abre o painel no navegador
"""

import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PASSOS = [
    ("1/3 Preparando os dados", [sys.executable, "src/preparar_dados.py"]),
    ("2/3 Treinando e avaliando", [sys.executable, "src/treinar.py"]),
    ("3/3 Rodando os testes", [sys.executable, "-m", "pytest", "-q"]),
]

for titulo, comando in PASSOS:
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")
    if subprocess.run(comando, cwd=RAIZ).returncode != 0:
        sys.exit(f"\nFalhou em: {titulo}. Veja a mensagem acima.")

print("\nTudo certo. Modelo em models/, gráficos em relatorio/.")
if "--app" in sys.argv:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"], cwd=RAIZ)
else:
    print("Para abrir o painel: streamlit run app/app.py")
