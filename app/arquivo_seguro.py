"""Gravação de arquivos de dados com nova tentativa e mensagem amigável.

No Windows, trocar um arquivo (`replace`) falha enquanto ele estiver aberto em
outro programa (editor, antivírus, sincronização de nuvem). Aqui cada troca tenta
de novo algumas vezes, com pausas curtas. Se continuar falhando, levanta
`ArquivoBloqueado` com um texto para a tela, e a API responde 409 (ver main.py),
em vez de devolver um erro 500 cru.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

TENTATIVAS = 3
PAUSA = 0.25          # segundos; a pausa cresce a cada tentativa


class ArquivoBloqueado(RuntimeError):
    """Não deu para gravar ou trocar o arquivo. A mensagem vai para a tela."""


def _com_tentativas(operacao: Callable[[], None], arquivo: Path) -> None:
    """Executa `operacao` até TENTATIVAS vezes; `arquivo` é o que o usuário deve fechar."""
    ultimo: OSError | None = None
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            operacao()
            return
        except OSError as exc:      # inclui PermissionError
            ultimo = exc
            if tentativa < TENTATIVAS:
                time.sleep(PAUSA * tentativa)
    raise ArquivoBloqueado(
        f"Não consegui gravar {arquivo.name}. Feche o arquivo {arquivo.name} se ele "
        "estiver aberto em outro programa (editor, antivírus ou sincronização) "
        "e tente de novo."
    ) from ultimo


def gravar_texto(destino: Path, texto: str, encoding: str = "utf-8") -> None:
    """Grava `texto` em `destino` de forma atômica (temporário + troca).

    Raises:
        ArquivoBloqueado: se a gravação falhar em todas as tentativas.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")

    def _gravar() -> None:
        tmp.write_text(texto, encoding=encoding)
        tmp.replace(destino)

    try:
        _com_tentativas(_gravar, destino)
    except ArquivoBloqueado:
        tmp.unlink(missing_ok=True)    # não deixa temporário para trás
        raise


def trocar(origem: Path, destino: Path, aberto: Path | None = None) -> None:
    """Move `origem` para `destino` (`replace`: sobrescreve se o destino existir).

    Quem chama deve escolher um `destino` livre, para não perder um arquivo antigo.

    Args:
        origem: Arquivo a mover.
        destino: Novo caminho.
        aberto: Arquivo que o usuário precisa fechar se der erro (padrão: `destino`).

    Raises:
        ArquivoBloqueado: se a troca falhar em todas as tentativas.
    """
    _com_tentativas(lambda: origem.replace(destino), aberto or destino)
