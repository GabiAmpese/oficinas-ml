"""Testes da preparação: limpeza, exclusões, intervalos e ausência de vazamento."""

import pandas as pd
import pytest

from preparar_dados import CAMINHO_INTERVALOS, CAMINHO_VISITAS, eh_troca_oleo, gerar_intervalos, unificar


def visitas_de(linhas):
    cols = ["fonte", "oficina", "codigo", "marca", "modelo", "ano", "motor", "combustivel", "data", "leitura", "servico",
            "tipo_oleo", "viscosidade", "intervalo_rec_unid", "intervalo_rec_meses", "uso", "aplicativo", "garagem",
            "cidade", "voltou_no_prazo", "observacoes"]
    base = {c: None for c in cols}
    return pd.DataFrame([{**base, **l} for l in linhas])


def test_regra_de_servico():
    assert eh_troca_oleo("Troca de óleo")
    assert eh_troca_oleo("Troca de óleo + filtros")
    assert eh_troca_oleo("troca de oleo + filtro")
    assert not eh_troca_oleo("Troca de filtro de óleo")
    assert not eh_troca_oleo("Freios")
    assert not eh_troca_oleo(None)


def test_lista_de_exclusao_funciona(monkeypatch):
    import preparar_dados
    monkeypatch.setattr(preparar_dados, "EXCLUIR", {"X_V9": "teste"})
    bruto = visitas_de([
        {"fonte": "X", "codigo": "V9", "data": "2025-01-01", "leitura": 100, "servico": "Troca de óleo"},
        {"fonte": "X", "codigo": "V1", "data": "2025-01-01", "leitura": 100, "servico": "Troca de óleo"},
    ])
    assert set(unificar(bruto)["veiculo"]) == {"X_V1"}


def test_unidade_horas_para_prefeitura_e_frota():
    bruto = visitas_de([
        {"fonte": "PREFEITURA", "codigo": "F1", "data": "2026-01-01", "leitura": 100, "servico": "Troca de óleo + filtros"},
        {"fonte": "CIROCAR", "codigo": "V1", "data": "2026-01-01", "leitura": 100, "servico": "Outro", "observacoes": "Carro de frota de empresa"},
        {"fonte": "CIROCAR", "codigo": "V2", "data": "2026-01-01", "leitura": 100, "servico": "Outro"},
    ])
    v = unificar(bruto).set_index("veiculo")
    assert v.loc["PREFEITURA_F1", "unidade"] == "h" and v.loc["PREFEITURA_F1", "frota"] == "Sim"
    assert v.loc["CIROCAR_V1", "unidade"] == "km" and v.loc["CIROCAR_V1", "frota"] == "Sim"
    assert v.loc["CIROCAR_V2", "frota"] == "Não"


def test_intervalos_valido_perdido_e_erro():
    bruto = visitas_de([
        {"fonte": "X", "codigo": "A", "data": "2025-01-01", "leitura": 10000, "servico": "Troca de óleo", "tipo_oleo": "Sintético"},
        {"fonte": "X", "codigo": "A", "data": "2025-07-01", "leitura": 18000, "servico": "Troca de óleo", "tipo_oleo": "Sintético"},  # válido
        {"fonte": "X", "codigo": "A", "data": "2026-07-01", "leitura": 40000, "servico": "Troca de óleo", "tipo_oleo": "Sintético"},  # perdida (>15000)
        {"fonte": "X", "codigo": "A", "data": "2026-08-01", "leitura": 39000, "servico": "Troca de óleo", "tipo_oleo": "Sintético"},  # erro (km volta)
        {"fonte": "X", "codigo": "B", "data": "2023-01-01", "leitura": 1000, "servico": "Troca de óleo"},
        {"fonte": "X", "codigo": "B", "data": "2026-01-01", "leitura": 5000, "servico": "Troca de óleo"},                           # perdida (>730 dias)
    ])
    iv = gerar_intervalos(unificar(bruto))
    assert iv["situacao"].tolist() == ["valido", "perdida", "erro", "perdida"] or sorted(iv["situacao"]) == ["erro", "perdida", "perdida", "valido"]
    valido = iv[iv["situacao"] == "valido"].iloc[0]
    assert valido["dias"] == 181 and valido["delta_unid"] == 8000
    # padrão do fabricante para sintético sem intervalo informado
    assert valido["intervalo_rec_unid"] == 10000 and valido["intervalo_rec_meses"] == 12


def test_features_sem_vazamento():
    """taxa_uso_ant e intervalo_ant só podem usar o que existia até data_ini."""
    bruto = visitas_de([
        {"fonte": "X", "codigo": "A", "data": "2025-01-01", "leitura": 0, "servico": "Troca de óleo"},
        {"fonte": "X", "codigo": "A", "data": "2025-04-01", "leitura": 9000, "servico": "Troca de óleo"},   # 90 dias, 100 km/dia
        {"fonte": "X", "codigo": "A", "data": "2025-07-10", "leitura": 10000, "servico": "Troca de óleo"},  # 100 dias, 10 km/dia
    ])
    iv = gerar_intervalos(unificar(bruto))
    primeiro, segundo = iv.iloc[0], iv.iloc[1]
    assert pd.isna(primeiro["taxa_uso_ant"]) and pd.isna(primeiro["intervalo_ant_dias"])
    assert segundo["taxa_uso_ant"] == pytest.approx(100.0)      # só conhece o 1º trecho, não o 2º
    assert segundo["intervalo_ant_dias"] == 90
    assert segundo["dias_regra"] == pytest.approx(min(10000 / 100.0, 12 * 30.4))


def test_arquivos_gerados_existem_e_batem():
    assert CAMINHO_VISITAS.exists() and CAMINHO_INTERVALOS.exists(), "rode python src/preparar_dados.py"
    iv = pd.read_csv(CAMINHO_INTERVALOS)
    v = iv[iv["situacao"] == "valido"]
    assert len(v) >= 50
    assert (v["dias"] > 0).all() and (v["delta_unid"] > 0).all() and (v["dias"] <= 730).all()
