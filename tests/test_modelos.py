"""Testes das réguas, do pipeline, do artefato e da decisão."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from modelos import (ReguaFabricante, ReguaMediana, ReguaUltimoIntervalo, construir_modelo, custo_antecedencia,
                     melhor_antecedencia, metricas, residuo_tipico, simular)
from treinar import CAMINHO_INTERVALOS, CAMINHO_METRICAS, CAMINHO_MODELO


def tabela():
    return pd.DataFrame({
        "fonte": ["A", "A", "B", "B"],
        "taxa_uso_ant": [100.0, np.nan, 10.0, 20.0],
        "intervalo_ant_dias": [50.0, np.nan, 200.0, np.nan],
        "intervalo_ant_unid": [5000.0, np.nan, 2000.0, np.nan],
        "n_trocas_ant": [2, 1, 3, 1],
        "idade_veic": [5, 10, 3, 8],
        "dias_regra": [100.0, np.nan, 365.0, 365.0],
        "intervalo_rec_unid": [10000.0, 10000.0, 10000.0, 10000.0],
        "intervalo_rec_meses": [12.0, 12.0, 12.0, 12.0],
        "tipo_oleo": ["Sintético", "Mineral", "Sintético", None],
        "unidade": ["km", "km", "km", "km"],
        "uso": ["Cidade", None, "Terra", "Misto"],
        "aplicativo": ["Não", "Sim", "Não", None],
        "frota": ["Não", "Não", "Sim", "Não"],
        "delta_unid": [5000.0, 6000.0, 3000.0, 4000.0],
    })


def test_regua_fabricante_calculada_a_mao():
    X = tabela(); y = pd.Series([90, 120, 300, 200])
    r = ReguaFabricante().fit(X, y)
    prev = r.predict(X)
    assert prev[0] == pytest.approx(min(10000 / 100, 12 * 30.4))     # 100 dias (km chega antes)
    assert prev[2] == pytest.approx(12 * 30.4)                        # 1000 dias por km, limitado pelos 12 meses
    # linha 1 sem taxa: usa a mediana da fonte A (100 km/dia) -> 100 dias
    assert prev[1] == pytest.approx(100.0)


def test_regua_fabricante_sem_taxa_na_unidade_usa_so_os_meses():
    X = tabela(); y = pd.Series([90, 120, 300, 200])
    r = ReguaFabricante().fit(X, y)
    maquina = pd.DataFrame([{**X.iloc[1].to_dict(), "fonte": "PREF", "unidade": "h", "taxa_uso_ant": np.nan,
                             "intervalo_rec_unid": 250.0, "intervalo_rec_meses": 6.0}])
    assert r.predict(maquina)[0] == pytest.approx(6 * 30.4)


def test_regua_ultimo_e_mediana():
    X = tabela(); y = pd.Series([90, 120, 300, 200])
    u = ReguaUltimoIntervalo().fit(X, y).predict(X)
    assert u[0] == 50 and u[2] == 200
    assert u[1] == pytest.approx(np.median([90, 120]))   # mediana da fonte A
    assert u[3] == pytest.approx(np.median([300, 200]))
    assert (ReguaMediana().fit(X, y).predict(X) == np.median(y)).all()


def test_pipeline_treina_e_aceita_categoria_desconhecida():
    X = tabela(); y = pd.Series([90, 120, 300, 200])
    for conjunto in ["reguas", "basico", "completo"]:
        m = construir_modelo("ridge", conjunto, alpha=1.0).fit(X, y)
        novo = X.iloc[[0]].copy(); novo["tipo_oleo"] = "Óleo de coco"; novo["fonte"] = "Z"
        prev = m.predict(novo)
        assert len(prev) == 1 and np.isfinite(prev[0])


def test_metricas_e_residuo():
    m = metricas([100, 200], [110, 180])
    assert m["mae_dias"] == 15 and m["n"] == 2
    assert residuo_tipico([100, 200, 300], [100, 200, 300]) == 0


def test_custo_antecedencia():
    real = np.array([100, 100, 100]); prev = np.array([110, 90, 160])
    # antecedência 0: lembrete em 110 (tarde), 90 (ok), 160 (tarde) -> 2 tarde; cedo: 100-160 = -60 (não)
    assert custo_antecedencia(real, prev, 0, 10, 1) == 20
    # antecedência 20: lembretes em 90, 70, 140 -> 1 tarde; cedo (>45 dias antes): 100-70=30 não
    assert custo_antecedencia(real, prev, 20, 10, 1) == 10
    a, custos = melhor_antecedencia(real, prev, 10, 1, antecedencias=range(0, 61))
    assert len(custos) == 61 and custos[a] == min(custos)


def test_artefato_carrega_preve_e_simula():
    assert CAMINHO_MODELO.exists(), "rode python src/treinar.py antes"
    art = joblib.load(CAMINHO_MODELO)
    iv = pd.read_csv(CAMINHO_INTERVALOS)
    caso = iv[iv["situacao"] == "valido"].iloc[[0]]
    for chave in ["modelo", "modelo_ml"]:
        p = art[chave].predict(caso)
        assert p.shape == (1,) and 0 < p[0] < 2000
    base = simular(art["modelo_ml"], caso)
    assert isinstance(base, float)


def test_split_temporal_sem_vazamento_no_registro():
    with open(CAMINHO_METRICAS, encoding="utf-8") as f:
        m = json.load(f)
    assert m["periodos"]["treino"][1] <= m["periodos"]["teste"][0]
    assert m["tamanhos"]["treino"] + m["tamanhos"]["teste"] >= 50
    assert "regua_fabricante" in m["validacao_cruzada"] and "teste_final" in m
    assert 0 <= m["faixa"]["cobertura_teste"] <= 1
