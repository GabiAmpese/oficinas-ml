"""Testes adicionais focados nos critérios de aceite e nas limitações do MVP.

Usar com ``python -m pytest tests/test_qualidade_adicional.py -q``.
Não modifica artefatos nem treina novamente o modelo.
"""

import numpy as np
import pandas as pd
import pytest

from modelos import ReguaFabricante, ReguaUltimoIntervalo
from treinar import dobras_temporais_estritas


def test_dobras_estritas_nao_usam_resultados_futuros():
    datas_fim = pd.date_range("2025-01-01", periods=25, freq="15D")
    # Intervalos de 45 dias tornam explícita a diferença entre data_ini e data_fim.
    X = pd.DataFrame({"data_ini": datas_fim - pd.Timedelta(days=45), "data_fim": datas_fim})
    dobras = dobras_temporais_estritas(X, n_blocos=5, minimo_treino=3)
    assert dobras, "A verificação deve produzir pelo menos uma dobra"
    for treino, validacao in dobras:
        assert len(treino) >= 3 and len(validacao) > 0
        assert set(treino).isdisjoint(set(validacao))
        assert X.iloc[treino]["data_fim"].max() <= X.iloc[validacao]["data_ini"].min()


def test_sem_historico_usa_limite_meses():
    X = pd.DataFrame({
        "fonte": ["OFICINA_TESTE"], "unidade": ["km"],
        "taxa_uso_ant": [np.nan], "intervalo_rec_unid": [10000.0],
        "intervalo_rec_meses": [12.0],
    })
    regra = ReguaFabricante().fit(X)
    assert regra.predict(X)[0] == pytest.approx(12 * 30.4)


def test_ultimo_intervalo_sem_historico_usa_mediana_da_fonte():
    X_treino = pd.DataFrame({"fonte": ["A", "A", "B"], "intervalo_ant_dias": [12.0, 15.0, 20.0]})
    y_treino = pd.Series([70.0, 130.0, 200.0])
    r = ReguaUltimoIntervalo().fit(X_treino, y_treino)
    X_novo = pd.DataFrame({"fonte": ["A", "C"], "intervalo_ant_dias": [np.nan, np.nan]})
    prev = r.predict(X_novo)
    assert prev[0] == pytest.approx(100.0)
    assert prev[1] == pytest.approx(130.0)
