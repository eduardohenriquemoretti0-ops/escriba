"""Memória de vozes: o Escriba aprende a reconhecer quem já falou antes.

Cada pessoa guarda uma lista de "contribuições", uma por reunião em que o
usuário confirmou quem ela era. Cada contribuição é a assinatura de voz
(vetor de timbre) daquela pessoa naquela reunião. A assinatura usada para
reconhecer é a média ponderada pelo tempo de fala, então quanto mais
reuniões confirmadas, mais estável fica o reconhecimento.

Regras:
- Só aprende o que o usuário CONFIRMOU na revisão. Nome deduzido sozinho
  não vira memória, para um erro não se reforçar.
- Confirmar de novo a mesma reunião substitui a contribuição anterior,
  em vez de somar duas vezes. Trocar o nome de um falante move a
  contribuição para a pessoa certa.
- Tudo fica só neste computador, em <dados>/vozes/vozes.json, e cada pessoa
  pode ser apagada pela tela "Vozes conhecidas". Assinatura de voz é dado
  biométrico (dado pessoal sensível na LGPD): grave e guarde só com
  consentimento dos participantes.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import arquivo_seguro, config

_lock = threading.RLock()

# Similaridade mínima (cosseno) para dizer "esta voz é de Fulano".
# Medido numa reunião real: mesma voz em trechos diferentes ≈ 0,97;
# vozes diferentes ≈ 0,21. Outra reunião (outro microfone/sala) derruba a
# similaridade da mesma pessoa; 0,60 deixa folga dos dois lados.
MATCH_MIN = 0.60
# A partir daqui o reconhecimento é tratado como "certeza".
MATCH_STRONG = 0.75
MAX_CONTRIBUTIONS = 60          # por pessoa (mantém as mais recentes)


def _file() -> Path:
    return config.DATA_DIR / "vozes" / "vozes.json"


def _model_id() -> str:
    return Path(config.DIARIZATION_MODEL).stem


def _destino_antigo(f: Path, modelo: str) -> Path:
    """Nome livre para guardar o arquivo de outro modelo (nunca sobrescreve)."""
    marca = re.sub(r"[^\w-]", "_", modelo) or "antigo"
    alvo = f.with_suffix(f".{marca}.json")
    if alvo.exists():
        alvo = f.with_suffix(f".{marca}.{int(time.time())}.json")
    return alvo


def _load() -> dict:
    f = _file()
    if not f.exists():
        return {"modelo": _model_id(), "pessoas": {}}
    data = json.loads(f.read_text(encoding="utf-8-sig"))
    if data.get("modelo") != _model_id():
        # Assinaturas de outro modelo não são comparáveis: começa do zero,
        # mas guarda o arquivo antigo em vez de apagar.
        arquivo_seguro.trocar(f, _destino_antigo(f, str(data.get("modelo", "antigo"))),
                              aberto=f)
        return {"modelo": _model_id(), "pessoas": {}}
    return data


def _save(data: dict) -> None:
    arquivo_seguro.gravar_texto(_file(), json.dumps(data, ensure_ascii=False))


def _signature(person: dict) -> Optional[np.ndarray]:
    contribs = person.get("contribuicoes") or []
    if not contribs:
        return None
    vecs = np.array([c["vetor"] for c in contribs], dtype=np.float32)
    w = np.array([max(c.get("segundos", 1.0), 1.0) for c in contribs], dtype=np.float32)
    v = (vecs * w[:, None]).sum(0)
    return v / (np.linalg.norm(v) + 1e-9)


# --------------------------------------------------------------------------
# Consulta
# --------------------------------------------------------------------------
def list_people() -> list[dict]:
    with _lock:
        data = _load()
    out = []
    for name, person in data["pessoas"].items():
        contribs = person.get("contribuicoes") or []
        out.append({
            "nome": name,
            "reunioes": len({c["job"] for c in contribs}),
            "minutos": round(sum(c.get("segundos", 0) for c in contribs) / 60, 1),
            "atualizado": max((c.get("quando", 0) for c in contribs), default=0),
        })
    return sorted(out, key=lambda p: p["nome"].casefold())


def match(centroids: dict, exclude: Optional[set] = None) -> dict:
    """Para cada voz da reunião, a pessoa conhecida mais parecida.

    `centroids`: {rótulo: vetor}. Devolve {rótulo: (nome, similaridade)}
    só para quem passou de MATCH_MIN; cada pessoa vai para no máximo uma voz
    (a mais parecida), porque uma pessoa não fala com duas vozes.
    """
    with _lock:
        data = _load()
    people = {n: _signature(p) for n, p in data["pessoas"].items()
              if n not in (exclude or set())}
    people = {n: v for n, v in people.items() if v is not None}
    if not people or not centroids:
        return {}
    pairs = []
    for label, vec in centroids.items():
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        for name, sig in people.items():
            pairs.append((float(v @ sig), label, name))
    result, used_labels, used_names = {}, set(), set()
    for sim, label, name in sorted(pairs, reverse=True):
        if sim < MATCH_MIN:
            break
        if label in used_labels or name in used_names:
            continue
        result[label] = (name, sim)
        used_labels.add(label)
        used_names.add(name)
    return result


# --------------------------------------------------------------------------
# Aprendizado
# --------------------------------------------------------------------------
def learn(job_id: str, assignments: dict, voices: dict) -> list[str]:
    """Grava o que o usuário confirmou numa reunião.

    `assignments`: {rótulo_cru: nome confirmado} (rótulo sem nome = esquecer).
    `voices`: {rótulo_cru: {"vetor": [...], "segundos": float}} da reunião.
    Devolve os nomes que foram aprendidos/atualizados.
    """
    learned = []
    with _lock:
        data = _load()
        people = data["pessoas"]
        # tira desta reunião tudo o que existia (renomes, nomes apagados)
        for person in people.values():
            person["contribuicoes"] = [
                c for c in person.get("contribuicoes", [])
                if not (c["job"] == job_id and c["rotulo"] in voices)]
        for label, name in assignments.items():
            voice = voices.get(label)
            if not name or not voice:
                continue
            person = people.setdefault(name, {"contribuicoes": []})
            person["contribuicoes"].append({
                "job": job_id, "rotulo": label, "vetor": voice["vetor"],
                "segundos": float(voice.get("segundos", 0.0)), "quando": time.time()})
            person["contribuicoes"] = person["contribuicoes"][-MAX_CONTRIBUTIONS:]
            if name not in learned:
                learned.append(name)
        data["pessoas"] = {n: p for n, p in people.items() if p["contribuicoes"]}
        _save(data)
    return learned


def forget(name: str) -> bool:
    with _lock:
        data = _load()
        if name not in data["pessoas"]:
            return False
        del data["pessoas"][name]
        _save(data)
        return True
