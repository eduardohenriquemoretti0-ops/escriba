"""Teste da tela (opcional): dirige o Escriba num Chromium de verdade.

Requer:  pip install playwright && playwright install chromium
Cobre: envio com nomes, cadastro de assinatura desenhada no canvas, revisão de
nomes/grafia, envio de e-mail (servidor SMTP local) e download do PDF.
"""
import os
import re
import threading
import time
from pathlib import Path
from unittest.mock import patch

import test_v2 as t          # define as variáveis de ambiente e os fakes
from aiosmtpd.controller import Controller
from playwright.sync_api import expect, sync_playwright

import uvicorn
from app import config
from app.main import app

SHOTS = Path(os.environ.get("ESCRIBA_TEST_OUT", t._TMP))
SHOTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    wav = Path(t._TMP) / "_ui.wav"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=25",
                    "-ac", "1", "-ar", "16000", str(wav)], check=True, capture_output=True)

    caixa = t.Caixa()
    smtp = Controller(caixa, hostname="127.0.0.1", port=t.SMTP_PORT)
    smtp.start()
    porta = t._porta_livre()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=porta, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    assert server.started, "servidor não subiu"
    url = f"http://127.0.0.1:{porta}"

    erros: list[str] = []
    try:
        with patch("app.engines.assemblyai_engine.AssemblyAIEngine.transcribe", t.fake_transcribe), \
             patch("app.summary.generate_ata", t.fake_ata), \
             sync_playwright() as pw:
            browser = pw.chromium.launch()
            ctx = browser.new_context(viewport={"width": 1100, "height": 900}, locale="pt-BR")
            page = ctx.new_page()
            page.on("pageerror", lambda e: erros.append(f"pageerror: {e}"))
            page.on("console", lambda m: m.type == "error" and "fonts.g" not in (m.location.get("url") or "")
                    and "ERR_" not in m.text
                    and "status of 422" not in m.text      # e-mail inválido (esperado)
                    and erros.append(f"console: {m.text}"))
            page.on("dialog", lambda d: d.accept())      # confirm()/alert()

            page.goto(url)
            expect(page.locator("#avisoAss")).to_be_visible()

            # ---- envio com nomes e ata ----------------------------------------
            # participantes em campos separados (v3)
            page.fill("#partUpload .part-linha:nth-child(1) input", "Rosilaine Andrade")
            page.click("#btnAddPartUpload")
            page.fill("#partUpload .part-linha:nth-child(2) input", "Eduardo Moretti")
            page.select_option("#engine", "assemblyai")   # motor simulado no teste
            page.check("#ata")
            page.set_input_files("#arquivo", str(wav))
            card = page.locator(".card").first
            expect(card.locator(".badge")).to_have_text("Concluído", timeout=30000)
            expect(card.get_by_role("button", name="Revisar nomes")).to_be_visible()
            expect(card.get_by_role("link", name="Ata PDF")).to_be_visible()
            expect(card.get_by_role("link", name="Transcrição PDF")).to_be_visible()
            page.screenshot(path=str(SHOTS / "ui_1_lista.png"), full_page=True)

            # ---- perfil + assinatura desenhada --------------------------------
            page.click("#btnPerfil")
            expect(page.locator("#dlgPerfil")).to_be_visible()
            page.fill("#pfNome", "Eduardo Moretti")
            page.fill("#pfCargo", "Diretor de Tecnologia")
            page.fill("#pfOrg", "Moretti Informática")
            page.fill("#pfEmail", "responsavel@exemplo.com")
            box = page.locator("#padCanvas").bounding_box()
            assert box and box["width"] > 200 and box["height"] > 100, box
            x0, y0 = box["x"] + 30, box["y"] + box["height"] * 0.6
            page.mouse.move(x0, y0)
            page.mouse.down()
            for i in range(1, 40):
                page.mouse.move(x0 + i * 7, y0 - 40 * ((i % 8) / 8) + (i % 3) * 6)
            page.mouse.up()
            page.screenshot(path=str(SHOTS / "ui_2_perfil.png"))
            page.click("#btnSalvarPerfil")
            expect(page.locator("#dlgPerfil")).to_be_hidden(timeout=10000)
            expect(page.locator("#toast")).to_contain_text("Perfil e assinatura salvos")
            expect(page.locator("#avisoAss")).to_be_hidden()
            prof = page.evaluate("fetch('/api/profile').then(r => r.json())")
            assert prof["tem_assinatura"] is True and prof["nome"] == "Eduardo Moretti", prof

            # reabre: assinatura atual aparece; envio de imagem (modo arquivo) funciona
            page.click("#btnPerfil")
            expect(page.locator("#assAtualBox")).to_be_visible()
            page.click("#abaArquivo")
            expect(page.locator("#painelArquivo")).to_be_visible()
            expect(page.locator("#painelDesenho")).to_be_hidden()
            sig = Path(t._TMP) / "foto.jpg"
            sig.write_bytes(t.foto_assinatura())
            page.set_input_files("#assArquivo", str(sig))
            expect(page.locator("#assPrevBox")).to_be_visible()
            page.click("#btnSalvarPerfil")
            expect(page.locator("#dlgPerfil")).to_be_hidden(timeout=10000)

            # ---- revisão de nomes -----------------------------------------------
            card.get_by_role("button", name="Revisar nomes").click()
            expect(page.locator("#dlgRev")).to_be_visible()
            parts = page.locator("#revPartLista input")
            expect(parts).to_have_count(2)
            expect(parts.nth(0)).to_have_value("Rosilaine Andrade")
            expect(parts.nth(1)).to_have_value("Eduardo Moretti")
            # nome deduzido da conversa ("aqui é a Rosilene") já vem sugerido
            expect(page.locator("#revFalantes .spk").first).to_contain_text("Sugerido: Rosilaine Andrade")
            expect(page.locator("#revFalantes .spk")).to_have_count(2)
            expect(page.locator("#revSug")).to_contain_text("Rosilene")
            page.locator("#revSug input[type=checkbox]").first.check()
            page.fill("#revFalantes input[data-label='Falante 1']", "Rosilaine Andrade")
            page.fill("#revFalantes input[data-label='Falante 2']", "Eduardo Moretti")
            page.fill("#revTitulo", "Reunião Acme")
            page.fill("#revData", "08/10/2026")
            page.fill("#revLocal", "Jundiaí")
            page.click("#btnAddCorr")
            page.locator("#revCorr .corr").last.locator(".de").fill("Acme")
            page.locator("#revCorr .corr").last.locator(".para").fill("ACME")
            page.screenshot(path=str(SHOTS / "ui_3_revisao.png"))
            page.click("#btnSalvarRev")
            expect(page.locator("#revEstado")).to_contain_text("Salvo", timeout=10000)
            expect(page.locator("#revSugBox")).to_be_hidden()      # sugestão virou correção
            expect(page.locator("#revCorr .corr")).to_have_count(2)
            page.click("#dlgRev [data-fechar].btn")
            expect(card.locator(".legenda")).to_contain_text("Rosilaine Andrade")
            txt = page.evaluate(f"fetch('/api/jobs/{card.locator('[data-id]').first.get_attribute('data-id')}/file/txt').then(r => r.text())")
            assert "Rosilaine Andrade · 00:00:00" in txt and "Rosilene" not in txt and "ACME" in txt, txt

            # ---- e-mail -----------------------------------------------------------
            card.get_by_role("button", name="Enviar por e-mail").click()
            expect(page.locator("#dlgEmail")).to_be_visible()
            expect(page.locator("#emNaoConfig")).to_be_hidden()
            expect(page.locator("#emAnexos input:checked")).to_have_count(2)
            expect(page.locator("#emMsg")).to_have_value(re.compile("Reunião Acme.*\\(08/10/2026\\)", re.S))
            page.fill("#emPara", "cliente@exemplo.com, outro@exemplo.com")
            page.screenshot(path=str(SHOTS / "ui_4_email.png"))
            page.click("#btnEnviarEmail")
            expect(page.locator("#emEstado")).to_have_text("Enviado.", timeout=30000)
            assert len(caixa.msgs) == 1, caixa.msgs
            nomes = sorted(a.get_filename() for a in caixa.msgs[0][2].iter_attachments())
            assert nomes == ["_ui_ata.pdf", "_ui_transcricao.pdf"], nomes
            expect(page.locator("#emHist li")).to_have_count(1)
            page.click("#dlgEmail [data-fechar].btn")

            # e-mail sem destinatário / inválido mostra erro legível
            card.get_by_role("button", name="Enviar por e-mail").click()
            page.fill("#emPara", "isso-nao-e-email")
            page.click("#btnEnviarEmail")
            expect(page.locator("#emEstado")).to_contain_text("inválido", timeout=10000)
            page.click("#dlgEmail [data-fechar].btn")

            # ---- download do PDF ----------------------------------------------------
            with page.expect_download() as dl:
                card.get_by_role("link", name="Ata PDF").click()
            destino = SHOTS / "ui_ata_baixada.pdf"
            dl.value.save_as(str(destino))
            assert destino.read_bytes()[:4] == b"%PDF" and dl.value.suggested_filename.endswith("_ata.pdf")

            # ---- refazer a ata (botão) ---------------------------------------------
            card.get_by_role("button", name="Refazer a ata").click()
            expect(card.get_by_role("button", name="Refazer a ata")).to_be_visible(timeout=15000)
            page.screenshot(path=str(SHOTS / "ui_5_final.png"), full_page=True)

            # ---- mobile -------------------------------------------------------------
            m = browser.new_context(viewport={"width": 390, "height": 800}, locale="pt-BR").new_page()
            m.goto(url)
            m.get_by_role("button", name="Revisar nomes").first.click()
            expect(m.locator("#dlgRev")).to_be_visible()
            largura = m.evaluate("document.documentElement.scrollWidth")
            assert largura <= 392, f"rolagem horizontal no celular: {largura}px"
            m.screenshot(path=str(SHOTS / "ui_6_mobile_revisao.png"))
            browser.close()
    finally:
        server.should_exit = True
        smtp.stop()

    assert not erros, "erros no navegador:\n" + "\n".join(erros)
    print("TESTE DA TELA PASSOU ✔  (capturas em", SHOTS, ")")


if __name__ == "__main__":
    main()
