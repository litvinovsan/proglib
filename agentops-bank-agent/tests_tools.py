"""Unit-тесты на edge cases инструментов агента — Проход 2, план блока A2.

Быстрые и детерминированные: НЕ дёргают живую модель (кроме одного
контролируемого mock на новый retry-механизм _structured_call, блок A1).
Реальное качество модели на edge cases — работа Evals (golden dataset,
решения №79-81): у unit-тестов другая задача — быстрая, воспроизводимая
проверка контракта инструментов, не качество генерации.

Запуск: python tests_tools.py   (или pytest tests_tools.py)
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

def _find_project_dir() -> Path:
    """Корень проекта от текущей папки вверх — тесты должны запускаться
    и из папки проекта, и из репозитория после clone."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "prompts").is_dir() and (candidate / "bench").is_dir():
            return candidate
    return here


PROJECT_DIR = _find_project_dir()
_ns = sys.modules["__main__"].__dict__


def _load_agent():
    nb = json.loads((PROJECT_DIR / "bank_support_agent.ipynb").read_text(encoding="utf-8"))
    n = 0
    for c in nb["cells"]:
        if c["cell_type"] == "code" and not c.get("id", "").endswith(("-verify", "-smoke")):
            n += 1
            exec("".join(c["source"]), _ns)
    print(f"=== агент загружен ({n} ячеек) ===\n")


_load_agent()

from pydantic import ValidationError


def _make_req(client_id="C-TEST", text="тест", request_id="TEST-REQ"):
    """Хелпер тестов. Имя НЕ _req: тесты исполняют ячейки ноутбука прямо
    в __main__, а там уже есть _req(state) — извлечение запроса из состояния.
    Совпадение имён молча подменяло бы функцию агента тестовой (решение №81)."""
    return _ns["IncomingRequest"](
        request_id=request_id, channel="app_chat",
        created_at=datetime.now(), client_id=client_id, text=text)


def test_empty_input_rejected_by_schema():
    """Пустой вход. IncomingRequest.text — min_length=1: граница доверия
    (решение №26) отсекает пустое обращение до того, как оно дойдёт
    до любого инструмента, а не где-то внутри узла."""
    try:
        _make_req(text="")
        raise AssertionError("пустой text обязан быть отвергнут схемой")
    except ValidationError:
        pass
    try:
        _make_req(text="   ")
        raise AssertionError("whitespace-only text обязан быть отвергнут (strip до проверки длины)")
    except ValidationError:
        pass


def test_oversized_input_rejected_by_schema():
    """Сверхдлинный вход. Ограничения длины в TicketInput/ReplyInput — не
    косметика, а барьер против протаскивания полотна текста инъекцией
    (см. докстринг TicketInput в ноутбуке)."""
    TicketInput = _ns["TicketInput"]
    ReplyInput = _ns["ReplyInput"]
    try:
        TicketInput(request_id="R1", client_id="C1", category="tariffs_fees",
                    priority="normal", summary="x" * 501)
        raise AssertionError("summary длиной 501 обязан быть отвергнут (max_length=500)")
    except ValidationError:
        pass
    try:
        ReplyInput(request_id="R1", client_id="C1", channel="app_chat",
                   category="tariffs_fees", complexity="simple", confidence="high",
                   text="x" * 4001)
        raise AssertionError("text длиной 4001 обязан быть отвергнут (max_length=4000)")
    except ValidationError:
        pass


def test_structured_call_retries_on_malformed_json():
    """Битый JSON от модели. _structured_call (блок A1) обязан переиграть
    вызов с объяснением причины, а не падать на первом сбое."""
    Classification = _ns["Classification"]
    _structured_call = _ns["_structured_call"]

    calls = {"n": 0}

    class _FakeMsg:
        def __init__(self, content): self.content = content

    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMsg(content)

    class _FakeResponse:
        def __init__(self, content): self.choices = [_FakeChoice(content)]

    good = Classification(summary="тест", tone="neutral", category="tariffs_fees",
                          subcategory="fee_dispute", complexity="simple",
                          priority="normal", confidence="high").model_dump_json()

    def _fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse("{это не json")   # первая попытка — брак
        return _FakeResponse(good)                  # вторая — валидный ответ

    real_client = _ns["client"]
    real_create = real_client.chat.completions.create
    real_client.chat.completions.create = _fake_create
    try:
        result = _structured_call(
            model="fake-model", temperature=0.0,
            messages=[{"role": "user", "content": "test"}],
            schema_cls=Classification)
        assert calls["n"] == 2, f"ожидался ровно один повтор, вызовов было {calls['n']}"
        assert result.category == "tariffs_fees"
    finally:
        real_client.chat.completions.create = real_create


def test_structured_call_raises_after_exhausting_retries():
    """Если модель ломает JSON на КАЖДОЙ попытке — обязана быть чистая
    типизированная ошибка, а не бесконечный цикл и не тихий возврат мусора."""
    Classification = _ns["Classification"]
    _structured_call = _ns["_structured_call"]
    StructuredCallError = _ns["StructuredCallError"]

    class _FakeMsg:
        def __init__(self, content): self.content = content

    class _FakeChoice:
        def __init__(self, content): self.message = _FakeMsg(content)

    class _FakeResponse:
        def __init__(self, content): self.choices = [_FakeChoice(content)]

    def _always_broken(**kwargs):
        return _FakeResponse("{всегда брак")

    real_client = _ns["client"]
    real_create = real_client.chat.completions.create
    real_client.chat.completions.create = _always_broken
    try:
        try:
            _structured_call(model="fake-model", temperature=0.0,
                             messages=[{"role": "user", "content": "test"}],
                             schema_cls=Classification, max_retries=1)
            raise AssertionError("обязана быть StructuredCallError после исчерпания попыток")
        except StructuredCallError:
            pass
    finally:
        real_client.chat.completions.create = real_create


def test_unknown_client_id_degrades_gracefully():
    """Несуществующий client_id. CRM read-only — отсутствие профиля не
    ошибка обработки, а неполнота контекста (докстринг enrich_client_context)."""
    enrich_client_context = _ns["enrich_client_context"]
    result = enrich_client_context("NOEXIST-CLIENT-999")
    assert result.status == "degraded"
    assert result.client_id == "NOEXIST-CLIENT-999"


def test_crm_unavailable_degrades_gracefully():
    """Недоступный CRM (не «не найден», а сбой доступа) — отдельная ветка
    except Exception в enrich_client_context, отдельный путь от KeyError."""
    enrich_client_context = _ns["enrich_client_context"]
    real_crm = _ns["CRM_DB"]

    class _FlakyCRM(dict):
        def __getitem__(self, key):
            if key == "CRM-DOWN-TEST":
                raise ConnectionError("симуляция сетевого сбоя CRM")
            return super().__getitem__(key)

    _ns["CRM_DB"] = _FlakyCRM(real_crm)
    try:
        result = enrich_client_context("CRM-DOWN-TEST")
        assert result.status == "degraded"
        assert "недоступна" in (result.error_message or "")
    finally:
        _ns["CRM_DB"] = real_crm


def test_create_ticket_idempotent_on_repeat():
    """Повтор create_ticket. Требование ТЗ: сетевой сбой после успешного
    создания не должен завести клиенту второй тикет при повторе узла."""
    TicketInput = _ns["TicketInput"]
    create_ticket = _ns["create_ticket"]

    inp = TicketInput(request_id="TEST-IDEM-UNIT", client_id="C-TEST",
                      category="tariffs_fees", subcategory="fee_dispute",
                      priority="normal", summary="проверка идемпотентности")
    first = create_ticket(inp)
    second = create_ticket(inp)
    assert first.ticket_id == second.ticket_id, "повтор обязан вернуть ТОТ ЖЕ ticket_id"
    assert not first.is_duplicate
    assert second.is_duplicate


def test_send_reply_refuses_forbidden_category():
    """send_reply с запрещённой категорией. Logic Engine (решение №28):
    даже если граф ошибётся в маршруте, инструмент откажет сам."""
    ReplyInput = _ns["ReplyInput"]
    send_reply = _ns["send_reply"]

    inp = ReplyInput(request_id="TEST-FORBIDDEN-CAT", client_id="C-TEST",
                     channel="app_chat", category="fraud_security",
                     complexity="simple", confidence="high",
                     text="текст ответа", sources=["doc01#001"])
    result = send_reply(inp)
    assert result.status == "error"
    assert result.refusal_reason == "category_not_allowed"


def test_hitl_pause_does_not_trip_time_fuse():
    """Пауза оператора не входит в бюджет времени агента.

    MAX_SECONDS (решение №89) ловит зависший вызов модели. HITL-пауза по
    замыслу длится сколько угодно — на то и чекпойнтер. До правки любое
    решение оператора позже 5 минут роняло инвариант времени в finalize
    уже ПОСЛЕ отправки ответа клиенту: исход перезаписывался на escalated
    без escalation_id. Тест держит границу между этими двумя ситуациями.
    """
    node_human_review = _ns["node_human_review"]
    node_finalize = _ns["node_finalize"]
    MAX_SECONDS = _ns["MAX_SECONDS"]

    state = {
        "request": _make_req(request_id="TEST-HITL-TIME"),
        "client": _ns["ClientContext"](client_id="C-TEST", client_name="Тест", is_vip=True),
        "classification": _ns["Classification"](
            summary="вопрос по ставке вклада", tone="neutral", category="deposits",
            subcategory="deposit_rate", complexity="simple", priority="normal",
            confidence="high"),
        "route": "send",
        "ticket_id": "TCK-HITL-TIME",
        "citations": ["doc01#001"],
        "step_count": 10,
        # оператор думал дольше потолка — штатный сценарий, не патология
        "started_at": time.monotonic() - (MAX_SECONDS + 60),
    }

    real_interrupt = _ns["interrupt"]
    _ns["interrupt"] = lambda payload: {"approved": True, "edited_text": None}
    try:
        delta = node_human_review(state)
    finally:
        _ns["interrupt"] = real_interrupt

    assert "started_at" in delta, "возобновление обязано перезапустить часы предохранителя"
    merged = {**state, **delta, "outcome": "auto_resolved"}
    trace = node_finalize(merged)["trace"][0]
    assert "НАРУШЕНЫ ИНВАРИАНТЫ" not in trace, f"ложное нарушение после паузы HITL: {trace}"

    # Обратная сторона: без сброса часов нарушение обязано срабатывать —
    # предохранитель не выключен, он просто не считает время человека.
    stale = node_finalize({**state, "outcome": "auto_resolved"})
    assert "превышен потолок времени" in stale["trace"][0], (
        "предохранитель по времени обязан срабатывать на зависшем пути")


def test_verify_skipped_only_when_it_cannot_matter():
    """Развилка 2: пропуск check_grounding при недостаточной поддержке.

    Проверяем не сам пропуск, а его БЕЗОПАСНОСТЬ: даже самый благоприятный
    ответ судьи (grounded=True) не может дать автоответ, если источников
    не хватает — значит вычисление доказуемо не влияет на исход, и
    пропускать его можно без изменения поведения агента.
    """
    route_after_draft = _ns["route_after_draft"]
    should_auto_reply = _ns["should_auto_reply"]
    Chunk, SearchHit = _ns["Chunk"], _ns["SearchHit"]

    def _hit(score):
        return SearchHit(chunk=Chunk(chunk_id="c1", text="текст фрагмента",
                                     document="doc01.docx", product="deposits",
                                     version="ВКЛАДЫ-2026", section="1. Ставки"),
                         score=score, found_by="dense")

    weak, strong = _hit(0.01), _hit(0.90)
    assert route_after_draft({"hits": [weak]}) == "decide"
    assert route_after_draft({"hits": []}) == "decide"
    assert route_after_draft({"hits": [strong]}) == "verify"

    base = {
        "classification": _ns["Classification"](
            summary="вопрос по ставке вклада", tone="neutral", category="deposits",
            subcategory="deposit_rate", complexity="simple", priority="low",
            confidence="high"),
        "language_ok": True, "step_count": 5, "started_at": time.monotonic(),
    }
    assert not should_auto_reply({**base, "hits": [weak], "grounded": True}),         "при недостаточной поддержке автоответ невозможен даже при grounded=True"
    assert should_auto_reply({**base, "hits": [strong], "grounded": True}),         "при достаточной поддержке и подтверждённом черновике автоответ обязан быть разрешён"


def test_significant_numbers_are_exact_not_rounded():
    """Числа сравниваются точно, а не с точностью формата.

    Здесь стояло f-форматирование по %g — шесть значащих цифр. Семизначная
    сумма и выдуманная соседняя давали один ключ, то есть барьер их не
    различал. В базе знаний 8 величин от миллиона (лимиты кредитов до 5, 7
    и 50 млн) — ровно те числа, ошибка в которых дороже всего.
    """
    sig = _ns["_significant_numbers"]

    assert sig("1 234 567 руб") != sig("1 234 566 руб"), (
        "соседние семизначные суммы обязаны различаться")
    assert sig("1 500 руб") == sig("1500 руб") == sig("01500 руб"), (
        "формат записи не должен влиять на сравнение")
    assert sig("7,50%") == sig("7.5%"), "запятая и точка — одно число"

    # Год и дата — не "значимое число": подтверждать их регламентом бессмысленно.
    assert sig("тариф действует с 2026 года") == set()
    assert sig("списание произошло 15.06.2026") == set()
    # Но ставка, записанная через точку, под шаблон даты попасть не должна.
    assert sig("ставка 0.5% годовых") == {"0.5"}


def test_grounding_blocks_fabricated_amount_without_model():
    """Первая ступень check_grounding ловит выдуманное число сама.

    Модель здесь не вызывается вообще: проверка чисел стоит до неё и
    возвращает результат сразу. Самый дешёвый слой ловит самое дорогое —
    неверную сумму в тексте, уходящем клиенту.
    """
    check_grounding = _ns["check_grounding"]
    Chunk, SearchHit = _ns["Chunk"], _ns["SearchHit"]
    hits = [SearchHit(
        chunk=Chunk(chunk_id="doc03#001",
                    text="Потребительский кредит: сумма от 100 000 до 7 000 000 руб.",
                    document="03_credits.docx", product="credits",
                    version="КРЕДИТЫ-2026", section="2. Кредиты"),
        score=0.9, found_by="dense")]

    bad = check_grounding("Вам доступен кредит на сумму до 7 000 001 руб.",
                          hits, query="какая максимальная сумма кредита?")
    assert not bad.grounded, "выдуманная сумма обязана быть отвергнута"
    assert bad.unsupported_numbers == ["7000001"], bad.unsupported_numbers

    # Обратная сторона: сумма, названная САМИМ клиентом, выдумкой не является.
    own = _ns["_significant_numbers"]("с меня списали 4 500 руб")
    assert own == {"4500"}


def test_kill_switch_stops_reply_approved_by_operator():
    """Рубильник обязан останавливать и путь через оператора.

    Раньше kill switch стоял только в handle_request — то есть закрывал вход,
    но не выход через паузу HITL. Обращение, вставшее на interrupt() ДО
    отключения агента, возобновлялось и отправляло ответ клиенту уже после
    того, как рубильник дёрнули. Причём именно по согласованному оператором
    тексту: он проходит мимо пяти барьеров автономности первой же строкой.
    """
    ReplyInput, send_reply = _ns["ReplyInput"], _ns["send_reply"]
    AGENT_STATE = _ns["AGENT_STATE"]

    inp = ReplyInput(request_id="TEST-KILL-SWITCH", client_id="C-TEST",
                     channel="app_chat", category="deposits", complexity="simple",
                     confidence="high", text="Ставка по вкладу — 7,5% годовых.",
                     sources=["doc02#001"], approved_by_human=True)

    AGENT_STATE["enabled"] = False
    try:
        blocked = send_reply(inp)
    finally:
        AGENT_STATE["enabled"] = True
    assert blocked.status == "error", "при отключённом агенте отправка обязана быть отклонена"
    assert blocked.refusal_reason == "agent_disabled", blocked.refusal_reason
    assert not blocked.sent

    # Рубильник вернули — тот же ввод проходит. Барьер именно временный,
    # а не сломавший согласованный оператором путь навсегда.
    allowed = send_reply(inp)
    assert allowed.sent, "после возврата рубильника согласованный ответ обязан уйти"


_TESTS = [
    test_empty_input_rejected_by_schema,
    test_oversized_input_rejected_by_schema,
    test_structured_call_retries_on_malformed_json,
    test_structured_call_raises_after_exhausting_retries,
    test_unknown_client_id_degrades_gracefully,
    test_crm_unavailable_degrades_gracefully,
    test_create_ticket_idempotent_on_repeat,
    test_send_reply_refuses_forbidden_category,
    test_hitl_pause_does_not_trip_time_fuse,
    test_verify_skipped_only_when_it_cannot_matter,
    test_significant_numbers_are_exact_not_rounded,
    test_grounding_blocks_fabricated_amount_without_model,
    test_kill_switch_stops_reply_approved_by_operator,
]

if __name__ == "__main__":
    failed = 0
    for t in _TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as ex:
            failed += 1
            print(f"  FAIL  {t.__name__}: {ex}")
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} тестов прошли")
    if failed:
        sys.exit(1)
