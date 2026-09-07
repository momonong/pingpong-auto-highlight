from __future__ import annotations

import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/pingpong_highlight/static"
INDEX_PATH = STATIC / "index.html"
APP_PATH = STATIC / "app.js"
I18N_PATH = STATIC / "i18n.js"
STYLES_PATH = STATIC / "styles.css"

HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")
CATALOG_KEY_RE = re.compile(r'^\s*"([^"\\]+)"\s*:', re.MULTILINE)
CATALOG_ENTRY_RE = re.compile(
    r'^\s*"(?P<key>[^"\\]+)"\s*:\s*'
    r'\[(?P<zh>"(?:\\.|[^"\\])*")\s*,\s*'
    r'(?P<en>"(?:\\.|[^"\\])*")\]\s*,?\s*$',
    re.MULTILINE,
)
LITERAL_KEY_RE = re.compile(
    r'''["'`]([a-z][A-Za-z0-9]*(?:\.[A-Za-z0-9_-]+)+)["'`]'''
)
LOCALIZED_TEXT_ATTRIBUTES = {
    "aria-label": "data-i18n-aria-label",
    "placeholder": "data-i18n-placeholder",
    "title": "data-i18n-title",
    "content": "data-i18n-content",
    "alt": "data-i18n-alt",
}
TRANSLATION_MARKERS = {
    "data-i18n",
    "data-i18n-html",
    *LOCALIZED_TEXT_ATTRIBUTES.values(),
}
VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _text_content(fragment: str) -> str:
    parser = _TextExtractor()
    parser.feed(fragment)
    return _normalize_text("".join(parser.parts))


class _IndexParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[dict[str, object]] = []
        self.elements: list[dict[str, object]] = []
        self.text_nodes: list[tuple[int, str, list[dict[str, object]]]] = []
        self.localized_nodes: list[tuple[int, str, str]] = []
        self.scripts: list[tuple[int, dict[str, str], tuple[str, ...]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        frame: dict[str, object] = {
            "tag": tag,
            "attrs": attributes,
            "line": self.getpos()[0],
            "text": [],
        }
        self.stack.append(frame)
        self.elements.append(frame)
        if tag == "script":
            ancestors = tuple(str(item["tag"]) for item in self.stack[:-1])
            self.scripts.append((self.getpos()[0], attributes, ancestors))
        if tag in VOID_ELEMENTS:
            self._close_frame(len(self.stack) - 1)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_ELEMENTS:
            self._close_frame(len(self.stack) - 1)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                while len(self.stack) > index:
                    self._close_frame(len(self.stack) - 1)
                return

    def handle_data(self, data: str) -> None:
        if not data.strip():
            for frame in self.stack:
                frame["text"].append(data)  # type: ignore[union-attr]
            return
        snapshot = list(self.stack)
        self.text_nodes.append((self.getpos()[0], data, snapshot))
        for frame in self.stack:
            frame["text"].append(data)  # type: ignore[union-attr]

    def _close_frame(self, index: int) -> None:
        frame = self.stack.pop(index)
        attrs = frame["attrs"]
        assert isinstance(attrs, dict)
        marker = attrs.get("data-i18n") or attrs.get("data-i18n-html")
        if marker:
            text = "".join(frame["text"])  # type: ignore[arg-type]
            self.localized_nodes.append((int(frame["line"]), str(marker), text))


def _parse_index() -> _IndexParser:
    parser = _IndexParser()
    parser.feed(INDEX_PATH.read_text(encoding="utf-8"))
    return parser


def _catalog_source() -> str:
    source = I18N_PATH.read_text(encoding="utf-8")
    prefix = "const catalog = Object.freeze({"
    start = source.index(prefix) + len(prefix)
    end = source.index("\n  });", start)
    return source[start:end]


def _load_catalog() -> tuple[list[str], dict[str, tuple[str, str]]]:
    source = _catalog_source()
    declared_keys = CATALOG_KEY_RE.findall(source)
    matches = list(CATALOG_ENTRY_RE.finditer(source))
    assert len(matches) == len(declared_keys), "Every catalog entry must be a literal string pair"
    catalog = {
        match.group("key"): (
            json.loads(match.group("zh")),
            json.loads(match.group("en")),
        )
        for match in matches
    }
    return declared_keys, catalog


def test_language_toggle_and_script_order() -> None:
    parser = _parse_index()
    toggles = [
        element
        for element in parser.elements
        if element["attrs"].get("id") == "languageToggle"  # type: ignore[union-attr]
    ]
    assert len(toggles) == 1
    toggle = toggles[0]
    attrs = toggle["attrs"]
    assert isinstance(attrs, dict)
    assert toggle["tag"] == "button"
    assert attrs.get("type") == "button"
    assert attrs.get("role") == "switch"
    assert attrs.get("aria-checked") in {"true", "false"}
    assert attrs.get("aria-label")
    assert attrs.get("title")

    toggle_options = {
        (_normalize_text(text), str(frame["attrs"].get("lang", "")))
        for _, text, stack in parser.text_nodes
        if any(frame["attrs"].get("id") == "languageToggle" for frame in stack)
        for frame in stack[-1:]
        if _normalize_text(text)
    }
    assert ("中", "zh-Hant") in toggle_options
    assert ("EN", "en") in toggle_options

    script_sources = [attrs.get("src") for _, attrs, _ in parser.scripts]
    i18n_index = script_sources.index("/static/i18n.js?v=1.4.0")
    app_index = script_sources.index("/static/app.js?v=1.4.0")
    assert i18n_index < app_index, "i18n.js must initialize the global before app.js runs"
    assert "head" in parser.scripts[i18n_index][2]
    assert "defer" not in parser.scripts[i18n_index][1]
    assert parser.scripts[app_index][1].get("defer") == ""


def test_blocking_bootstrap_clears_pending_state_after_dom_is_ready() -> None:
    source = I18N_PATH.read_text(encoding="utf-8")
    styles = STYLES_PATH.read_text(encoding="utf-8")
    pending_assignment = 'document.documentElement.dataset.i18nPending = "true";'
    pending_cleanup = "delete document.documentElement.dataset.i18nPending;"
    ready_check = 'document.readyState === "loading"'
    dom_ready_fallback = 'document.addEventListener("DOMContentLoaded", apply, { once: true });'

    assert pending_assignment in source
    assert pending_cleanup in source
    assert source.index(pending_assignment) < source.index("function apply()")
    assert ready_check in source
    assert dom_ready_fallback in source
    assert re.search(r"else\s*\{\s*apply\(\);\s*\}", source)
    assert re.search(r"html\[data-i18n-pending\]\s+body\s*\{[^}]*visibility:\s*hidden", styles)


def test_traditional_chinese_static_copy_has_translation_markers() -> None:
    parser = _parse_index()
    unmarked: list[str] = []
    for line, text, stack in parser.text_nodes:
        if not HAN_RE.search(text):
            continue
        marked = any(
            any(marker in frame["attrs"] for marker in ("data-i18n", "data-i18n-html"))
            for frame in stack
        )
        language_autonym = any(
            frame["attrs"].get("id") == "languageToggle" for frame in stack
        ) and any(frame["attrs"].get("lang") == "zh-Hant" for frame in stack)
        if not marked and not language_autonym:
            unmarked.append(f"line {line}: {_normalize_text(text)!r}")
    assert not unmarked, "Unmarked Traditional Chinese text:\n" + "\n".join(unmarked)


def test_translatable_chinese_attributes_have_translation_markers() -> None:
    parser = _parse_index()
    unmarked: list[str] = []
    for element in parser.elements:
        attrs = element["attrs"]
        assert isinstance(attrs, dict)
        for attribute, marker in LOCALIZED_TEXT_ATTRIBUTES.items():
            value = str(attrs.get(attribute, ""))
            if not HAN_RE.search(value):
                continue
            stateful_language_toggle = (
                attrs.get("id") == "languageToggle" and attribute in {"aria-label", "title"}
            )
            if marker not in attrs and not stateful_language_toggle:
                unmarked.append(
                    f"line {element['line']}: <{element['tag']}> {attribute}={value!r}"
                )
    assert not unmarked, "Unmarked translatable attributes:\n" + "\n".join(unmarked)


def test_app_javascript_contains_no_chinese_copy() -> None:
    source = APP_PATH.read_text(encoding="utf-8").replace("、", "")
    matches = [
        f"line {line_number}: {line.strip()}"
        for line_number, line in enumerate(source.splitlines(), start=1)
        if HAN_RE.search(line)
    ]
    assert not matches, "Move Chinese UI copy from app.js into i18n.js:\n" + "\n".join(matches)


def test_all_literal_translation_keys_exist() -> None:
    _, catalog = _load_catalog()
    parser = _parse_index()
    html_keys = {
        str(value)
        for element in parser.elements
        for name, value in element["attrs"].items()  # type: ignore[union-attr]
        if name in TRANSLATION_MARKERS
    }
    app_keys = set(LITERAL_KEY_RE.findall(APP_PATH.read_text(encoding="utf-8")))
    required_toggle_keys = {
        "language.currentChinese",
        "language.currentEnglish",
        "language.switchChinese",
        "language.switchEnglish",
    }
    missing = sorted((html_keys | app_keys | required_toggle_keys) - catalog.keys())
    assert not missing, f"Translation keys missing from the catalog: {missing}"


def test_catalog_has_consistent_bilingual_values_and_placeholders() -> None:
    declared_keys, catalog = _load_catalog()
    duplicates = sorted(key for key, count in Counter(declared_keys).items() if count > 1)
    assert not duplicates, f"Duplicate translation keys: {duplicates}"
    assert set(declared_keys) == set(catalog)

    problems: list[str] = []
    for key, (traditional_chinese, english) in catalog.items():
        if not traditional_chinese.strip() or not english.strip():
            problems.append(f"{key}: both language values must be non-empty")
        if HAN_RE.search(english):
            problems.append(f"{key}: English value still contains Chinese characters")
        chinese_parameters = Counter(PLACEHOLDER_RE.findall(traditional_chinese))
        english_parameters = Counter(PLACEHOLDER_RE.findall(english))
        if chinese_parameters != english_parameters:
            problems.append(
                f"{key}: placeholder mismatch {dict(chinese_parameters)} != "
                f"{dict(english_parameters)}"
            )
    assert not problems, "Catalog consistency errors:\n" + "\n".join(problems)


def test_html_fallback_copy_matches_traditional_chinese_catalog_values() -> None:
    _, catalog = _load_catalog()
    parser = _parse_index()
    mismatches: list[str] = []

    for line, key, fallback in parser.localized_nodes:
        if key not in catalog:
            continue
        expected = _text_content(catalog[key][0])
        actual = _normalize_text(fallback)
        if actual != expected:
            mismatches.append(f"line {line}: {key}: {actual!r} != {expected!r}")

    for element in parser.elements:
        attrs = element["attrs"]
        assert isinstance(attrs, dict)
        for attribute, marker in LOCALIZED_TEXT_ATTRIBUTES.items():
            key = attrs.get(marker)
            if not key or key not in catalog:
                continue
            actual = _normalize_text(str(attrs.get(attribute, "")))
            expected = _normalize_text(catalog[str(key)][0])
            if actual != expected:
                mismatches.append(
                    f"line {element['line']}: {key}: {actual!r} != {expected!r}"
                )

    assert not mismatches, "HTML fallbacks do not match the zh-Hant catalog:\n" + "\n".join(
        mismatches
    )
