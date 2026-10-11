"""Salvamento de arquivos pela janela desktop (pywebview).

Por que existe: o download pelo QtWebEngine passa pelo handler
`on_download_requested` do pywebview, que abre um diálogo modal dentro do sinal
`downloadRequested`. Quando esse diálogo não é concluído, o arquivo simplesmente
não é gravado, e nada aparece para o usuário. Aqui a janela pede o caminho com
`create_file_dialog` (fora do QtWebEngine) e este módulo grava o arquivo.

O conteúdo sai da mesma função da rota HTTP (`app.main._file_for`), então TXT,
SRT, VTT, JSON, ata e PDFs continuam sendo gerados num lugar só.
"""
from __future__ import annotations

import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

from app import main as servidor

# Tipos que a janela pode salvar: os mesmos da rota HTTP de download.
KINDS_SALVAVEIS: frozenset[str] = frozenset(servidor._MIME) | frozenset(servidor._PDF_KINDS)

_JOB_ID = servidor._JOB_ID
_CARACTERES_PROIBIDOS = re.compile(r"[^\w\-. ()\[\]]")
_NOMES_RESERVADOS = {"CON", "PRN", "AUX", "NUL",
                     *(f"COM{i}" for i in range(1, 10)),
                     *(f"LPT{i}" for i in range(1, 10))}
_TAMANHO_MAX_NOME = 120


class SalvarError(Exception):
    """Erro esperado ao salvar. A mensagem é mostrada ao usuário."""


@dataclass(frozen=True)
class Arquivo:
    """Arquivo pronto para gravar: origem em disco e nome sugerido."""

    origem: Path
    extensao: str   # sem o ponto, ex.: "pdf"
    nome: str       # ex.: "reuniao_transcricao.pdf"


def nome_seguro(sugerido: str, extensao: str, padrao: str) -> str:
    """Devolve um nome de arquivo sem caminho, sem caracteres de controle.

    A extensão é sempre a `extensao` pedida pelo servidor, nunca a do texto
    recebido. Se `sugerido` não sobrar nada útil, usa `padrao`.

    Args:
        sugerido: Nome vindo da página (pode ser vazio ou malicioso).
        extensao: Extensão do arquivo real, sem o ponto.
        padrao: Nome de reserva, já com extensão.

    Returns:
        Nome do arquivo com extensão.
    """
    texto = unicodedata.normalize("NFC", str(sugerido or ""))
    texto = os.path.basename(texto.replace("\\", "/"))
    stem = Path(texto).stem if texto else ""
    stem = _CARACTERES_PROIBIDOS.sub("_", stem).strip(" .")[:_TAMANHO_MAX_NOME]
    if not stem:
        return padrao
    if stem.upper() in _NOMES_RESERVADOS:
        stem = f"_{stem}"
    return f"{stem}.{extensao}"


def preparar(job_id: str, kind: str) -> Arquivo:
    """Valida o pedido e gera (ou localiza) o arquivo a ser salvo.

    Args:
        job_id: Id do job (12 caracteres hexadecimais).
        kind: Tipo pedido, dentro de `KINDS_SALVAVEIS`.

    Returns:
        O arquivo de origem com o nome sugerido.

    Raises:
        SalvarError: Pedido inválido, job inexistente ou falha ao gerar.
    """
    if not isinstance(job_id, str) or not _JOB_ID.match(job_id):
        raise SalvarError("Identificador de transcrição inválido.")
    if kind not in KINDS_SALVAVEIS:
        raise SalvarError("Tipo de arquivo inválido.")
    try:
        job = servidor._get_job(job_id)
        origem, _mime, nome = servidor._file_for(job, kind)
    except HTTPException as exc:
        raise SalvarError(str(exc.detail)) from exc
    extensao = Path(nome).suffix.lstrip(".")
    return Arquivo(origem=origem, extensao=extensao, nome=nome)


def gravar(origem: Path, destino: Path) -> Path:
    """Copia `origem` para `destino` de forma atômica.

    Grava num arquivo temporário ao lado e só depois troca pelo nome final, então
    um erro no meio não deixa um arquivo pela metade com o nome certo.

    Args:
        origem: Arquivo já gerado.
        destino: Caminho escolhido pelo usuário.

    Returns:
        O caminho gravado.

    Raises:
        SalvarError: Destino inválido, pasta inexistente, sem permissão ou
            arquivo de origem vazio.
    """
    destino = Path(destino).expanduser()
    if destino.is_dir():
        raise SalvarError(f"O caminho escolhido é uma pasta: {destino}")
    if not destino.parent.is_dir():
        raise SalvarError(f"A pasta não existe: {destino.parent}")
    temporario = destino.with_name(f".{destino.name}.part")
    try:
        shutil.copyfile(origem, temporario)
        if temporario.stat().st_size == 0:
            raise SalvarError("O arquivo gerado está vazio; nada foi gravado.")
        os.replace(temporario, destino)
    except OSError as exc:
        raise SalvarError(
            f"Não consegui gravar em {destino}: {exc.strerror or exc}") from exc
    finally:
        temporario.unlink(missing_ok=True)   # só sobra se houve erro
    return destino


def salvar_em(job_id: str, kind: str, destino: Path) -> Path:
    """Gera o arquivo do job e grava no `destino` (sem diálogo).

    É o que a janela chama depois de o usuário escolher o caminho, e também o
    que os testes usam.

    Args:
        job_id: Id do job.
        kind: Tipo pedido.
        destino: Caminho final do arquivo.

    Returns:
        O caminho gravado.

    Raises:
        SalvarError: Qualquer falha esperada de validação, geração ou gravação.
    """
    arquivo = preparar(job_id, kind)
    return gravar(arquivo.origem, destino)
