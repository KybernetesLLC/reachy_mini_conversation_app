"""The output gate: what of a reply's transcript is spoken (2026-10-16, 2026-10-17).

One place decides whether a reply plays. Its rules run in order over the transcript so
far, each returning a typed Verdict or None (no opinion), and the first decisive one
wins:

1. leaked_tool_name: the reply is only a registered tool's name, bare or in call markup.
2. artifact_shape: the reply is only stage directions, call markup or thinking.

The handler holds audio while the verdict is undecided, cuts on CUT, and at the end of
the transcript (final=True) asks once more, so a bare name is judged on the whole reply.
The supervisor keeps no rule of its own: the transcript notification carries the cut's
kind, and a marked reply is never stored.
"""

from __future__ import annotations
import re
import json
from typing import Final, Callable, Iterable
from dataclasses import dataclass


# The audio cut (2026-10-16): a reply whose transcript is only stage directions
# ("(stops talking)", "*sighs*"), tool-call markup leaked into text ("call:stay_quiet{}",
# "<call:stay_quiet/>"), or thinking ("<think>...</think>"), plus whitespace and stray
# ". ! , ;", is never spoken. A reply with any real words outside those shapes ("(Laughs)
# yes, it leaves at 3") is spoken whole: losing the answer is worse (the owner,
# 2026-10-16). The shapes cover Gemma 4 (call:name{...}, <|tool_call>...<tool_call|>,
# malformed <call:name/>, stray special tokens), Qwen3 (<tool_call>{json}</tool_call>),
# Qwen3.5 / Qwen3-Coder (<function=name>...</function>), Gemma 3 ([name(a=1)], a
# ```tool_code fence) and generic forms (<function_call>, a JSON {"name", "arguments"}
# object). speech-to-speech sends each text part's transcript delta ahead of its audio,
# so the first deltas decide before any audio arrives; while the text so far could still
# become an artifact (an open bracket, "<" with no ">", a prefix of "call:", an open
# <think>) the verdict is "undecided", but only under _UNDECIDED_MAX characters. Audio
# that arrives first is held for at most SPOKEN_ARTIFACT_HOLD_S, then released. The
# server's copy of the reply is left alone.
_UNDECIDED: Final[str] = "undecided"
_STAGE: Final[str] = "stage direction"
_CALL: Final[str] = "leaked tool call"
_THINKING: Final[str] = "thinking"
_DIRECTION_MAX = 80  # characters inside a stage direction's brackets
_UNDECIDED_MAX = 200  # past this much text, an unfinished shape is a reply
_GEMMA_QUOTE: Final[str] = '<|"|>'
_DIRECTION_CLOSERS: Final[dict[str, str]] = {"(": ")", "[": "]", "*": "*", "_": "_"}
_PAIRS: Final[dict[str, str]] = {"(": ")", "[": "]", "{": "}"}
# Whole special tokens and stray closing tags: <|"|>, <|im_end|>, <|tool_call>, <tool_call|>.
_TOKEN = re.compile(
    r"<\|[^<>|\s]{1,40}\|>|<\|\w{1,40}>|<\w{1,40}\|>"
    r"|</(?:tool_call|function_call|function|think|thinking|call)\s*>",
    re.IGNORECASE,
)
# Wrappers whose whole body is markup: (opener, closer, kind).
_WRAPPERS = (
    (re.compile(r"<(think|thinking)>", re.IGNORECASE), "</{}>", _THINKING),
    (re.compile(r"<(tool_call|function_call)>", re.IGNORECASE), "</{}>", _CALL),
    (re.compile(r"<function=[\w.-]+>", re.IGNORECASE), "</function>", _CALL),
)
_TAG_CALL = re.compile(r"<call:\s*[\w.-]+\s*", re.IGNORECASE)
_TAG_CALL_END = re.compile(r"[^<>]*?(/?)>")
_TAG_CALL_CLOSER = re.compile(r"</call(?::[\w.-]*)?\s*>", re.IGNORECASE)
_BARE_CALL = re.compile(r"call:\s*[\w.-]+\s*", re.IGNORECASE)
_BARE_CALL_OPEN = re.compile(r"call:\s*[\w.-]*\s*", re.IGNORECASE)
_CALL_ITEM = r"[A-Za-z_][\w.]*\(.*\)"
_CALL_LIST = re.compile(rf"\[\s*{_CALL_ITEM}(?:\s*,\s*{_CALL_ITEM})*\s*\]", re.DOTALL)
_TOOL_CODE = re.compile(r"```\s*tool_code\b.*?```", re.IGNORECASE | re.DOTALL)
_JSON_CALL_KEYS: Final[frozenset[str]] = frozenset({"name", "arguments", "parameters"})


def _skip(text: str, i: int) -> int:
    """Return the index past any whitespace and stray ". ! , ;" at text[i]."""
    return i + len(text[i:]) - len(text[i:].lstrip(" \t\r\n.!,;"))


def _balanced_end(text: str, i: int) -> int:
    """Return the index past the bracket closing text[i] ("(", "[" or "{"), or -1 while open.

    Nested brackets count; "..." and Gemma 4 <|"|>...<|"|> strings are skipped.
    """
    stack = [_PAIRS[text[i]]]
    j = i + 1
    while j < len(text):
        if text.startswith(_GEMMA_QUOTE, j):
            quote_end = text.find(_GEMMA_QUOTE, j + len(_GEMMA_QUOTE))
            if quote_end == -1:
                return -1
            j = quote_end + len(_GEMMA_QUOTE)
            continue
        c = text[j]
        if c == '"':
            j += 1
            while j < len(text) and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            if j >= len(text):
                return -1
        elif c in _PAIRS:
            stack.append(_PAIRS[c])
        elif c == stack[-1]:
            stack.pop()
            if not stack:
                return j + 1
        j += 1
    return -1


def _json_call(body: str) -> bool:
    """Whether a JSON object is only a tool call: {"name": ..., "arguments"|"parameters": ...}."""
    try:
        value = json.loads(body)
    except ValueError:
        return False
    return (
        isinstance(value, dict)
        and isinstance(value.get("name"), str)
        and ("arguments" in value or "parameters" in value)
        and set(value) <= _JSON_CALL_KEYS
    )


def _closed(kind: str, end: int) -> tuple[str, int]:
    """Return (kind, end) for a closed shape, or ("undecided", -1) when end is -1 (still open)."""
    return (kind, end) if end != -1 else (_UNDECIDED, -1)


def _tag_call_at(text: str, i: int, final: bool = False) -> tuple[str, int] | None:
    """Match a malformed Gemma 4 call at text[i]: <call:name/>, <call:name>...</call> and kin.

    At the end of the reply (`final`), a call whose arguments closed but whose tag never did
    ("<call:idle_do_nothing{reason:...}", 2026-10-20) ends at its arguments."""
    opener = _TAG_CALL.match(text, i)
    if not opener:
        return None
    j = opener.end()
    braced = j < len(text) and text[j] == "{"
    if braced:
        j = _balanced_end(text, j)
        if j == -1:
            return _UNDECIDED, -1
    tag_end = _TAG_CALL_END.match(text, j)
    if not tag_end:
        if final and braced:
            return _CALL, j
        return (_UNDECIDED, -1) if ">" not in text[j:] else None
    if tag_end.group(1):  # self-closing
        return _CALL, tag_end.end()
    closer = _TAG_CALL_CLOSER.search(text, tag_end.end())
    if closer:
        return _CALL, closer.end()
    k = _skip(text, tag_end.end())
    if k < len(text) and text[k] == "{":
        return _closed(_CALL, _balanced_end(text, k))
    return _CALL, tag_end.end()


def _angle_at(text: str, i: int, final: bool = False) -> tuple[str, int] | None:
    """Match a shape opening with "<" at text[i]: a special token, a wrapper or a tag call."""
    token = _TOKEN.match(text, i)
    if token:
        return _CALL, token.end()
    for opener, closer, kind in _WRAPPERS:
        found = opener.match(text, i)
        if found:
            close = closer.format(*found.groups()).lower()
            close_at = text.lower().find(close, found.end())
            return _closed(kind, close_at + len(close) if close_at != -1 else -1)
    tag = _tag_call_at(text, i, final)
    if tag:
        return tag
    return (_UNDECIDED, -1) if ">" not in text[i:] else None


def _shape_at(text: str, i: int, final: bool = False) -> tuple[str, int] | None:
    """Match the artifact at text[i]: (kind, end), ("undecided", -1) while open, or None."""
    rest = text[i:]
    c = text[i]
    if c == "<":
        return _angle_at(text, i, final)
    if rest[:5].lower() == "call:":
        call = _BARE_CALL.match(text, i)
        if call and call.end() < len(text) and text[call.end()] == "{":
            return _closed(_CALL, _balanced_end(text, call.end()))
        return (_UNDECIDED, -1) if _BARE_CALL_OPEN.fullmatch(rest) else None
    if "call:".startswith(rest.lower()):
        return _UNDECIDED, -1
    if c == "`":
        if not rest.startswith("```"):
            return (_UNDECIDED, -1) if "```".startswith(rest) else None
        fence = _TOOL_CODE.match(text, i)
        if fence:
            return _CALL, fence.end()
        return (_UNDECIDED, -1) if rest.find("```", 3) == -1 else None
    if c == "{":
        end = _balanced_end(text, i)
        if end == -1:
            return _UNDECIDED, -1
        return (_CALL, end) if _json_call(text[i:end]) else None
    if c == "[":
        end = _balanced_end(text, i)
        if end != -1 and _CALL_LIST.fullmatch(text, i, end):
            return _CALL, end
    if c in _DIRECTION_CLOSERS:
        close_at = text.find(_DIRECTION_CLOSERS[c], i + 1)
        if close_at == -1:
            return _UNDECIDED, -1
        inner = text[i + 1 : close_at]
        if inner.strip() and len(inner) <= _DIRECTION_MAX and not set(inner) & set("()[]*{}"):
            return _STAGE, close_at + 1
    return None


def _shapes(text: str, final: bool = False) -> list[tuple[str, str]] | str | None:
    """The artifact shapes `text` is made of, as (kind, the shape's own text);
    "undecided" while one is still open; None when there are real words outside them.
    """
    head = text.strip()
    found: list[tuple[str, str]] = []
    i = _skip(head, 0)
    while i < len(head):
        shape = _shape_at(head, i, final)
        if shape is None:
            return None
        kind, end = shape
        if kind == _UNDECIDED:
            return _UNDECIDED if len(head) <= _UNDECIDED_MAX else None
        found.append((kind, head[i:end]))
        i = _skip(head, end)
    return found if found else _UNDECIDED


def spoken_artifact(text: str, final: bool = False) -> str | None:
    """Return "stage direction", "leaked tool call", "thinking", None (a reply), or "undecided".

    Text made only of artifact shapes, whitespace and stray ". ! , ;" is an artifact,
    named by its first shape. Pure.
    """
    shapes = _shapes(text, final)
    return shapes[0][0] if isinstance(shapes, list) else shapes


def without_calls(text: str) -> str:
    """A played reply's text with its call markup taken out: what the supervisor keeps.

    The audio is untouched (a reply with real words plays whole, the owner, 2026-10-16),
    but kept markup reached his word-for-word memory and he copied it (2026-10-20). Pure.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        at_word_start = i == 0 or not text[i - 1].isalnum()
        if text[i] == "<" or (at_word_start and text[i : i + 5].lower() == "call:"):
            shape = _shape_at(text, i, final=True)
            if shape is not None and shape[0] == _CALL:
                i = shape[1]
                continue
        out.append(text[i])
        i += 1
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()


# --- the gate ------------------------------------------------------------------------

PLAY: Final[str] = "play"
CUT: Final[str] = "cut"
CUT_AND_EXECUTE: Final[str] = "cut and execute"
PLAY_AND_LOG: Final[str] = "play and log"
UNDECIDED: Final[str] = _UNDECIDED


@dataclass(frozen=True)
class Verdict:
    """What to do with a reply. `kind` names what a cut removes ("stage direction",
    "leaked tool call", "thinking"); `tool` and `shape` name a leaked tool and how it
    leaked ("markup", "underscore name", "spoken name", "plain word").
    """

    action: str
    kind: str | None = None
    tool: str | None = None
    shape: str | None = None


# The tool-name leak guard (the planning session, 2026-10-17): the hosted model said
# "stay_quiet" aloud at 20:44:56, heard 0.22 s later, and every leak in 10 days was
# stay_quiet or idle_do_nothing, in five shapes: "stay_quiet", "Stay quiet.",
# "call:stay_quiet{}", "<call:stay_quiet/>" and "<call:idle_do_nothing{reason:...}/>". The
# names come from the live registry at each session open, so a new tool is covered
# without a list to keep. On a whole reply: markup or an underscore name is cut; a
# multi-word name alone ("stay quiet") is cut; a single plain word ("Dance!") plays,
# logged as a possible leak; a sentence with a tool's word in it plays. A cut leak runs
# its tool only when the tool is `leak_safe` (stay_quiet, idle_do_nothing: doing nothing
# is what they do); every other leak is cut and logged, never run.
_MIN_WAIT = 3  # characters before the start of a reply can wait to become a name
_WORD_EDGES: Final[str] = " \t\r\n.!,;?:'\"`*"
_PLAIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\s-]{0,79}")
_CALLED_NAME = (
    re.compile(r"call:\s*([\w.-]+)", re.IGNORECASE),
    re.compile(r"<function=([\w.-]+)>", re.IGNORECASE),
    re.compile(r'"name"\s*:\s*"([^"]+)"'),
    re.compile(r"^\[\s*([A-Za-z_][\w.]*)\("),
    re.compile(r"tool_code\s*([A-Za-z_][\w.]*)\(", re.IGNORECASE),
)


def _normal(word: str) -> str:
    """Lowercase, with each run of spaces and hyphens an underscore."""
    return re.sub(r"[\s-]+", "_", word.strip().lower())


@dataclass(frozen=True)
class LeakTool:
    name: str
    leak_safe: bool


class ToolNames:
    """The registry's tool names, in every form a reply could leak them: the name and,
    for a namespaced one, its part after the last "__".
    """

    def __init__(self, tools: Iterable[tuple[str, bool]] = ()) -> None:
        self._forms: dict[str, LeakTool] = {}
        for name, leak_safe in tools:
            tool = LeakTool(name, leak_safe)
            for form in (name, name.rsplit("__", 1)[-1]):
                self._forms[_normal(form)] = tool

    def find(self, normal: str) -> LeakTool | None:
        return self._forms.get(normal)

    def could_become(self, normal: str) -> bool:
        """Whether more words could make this a multi-word name ("stay" -> "stay quiet").

        Under _MIN_WAIT characters it never waits: "I" starts 133 of 659 replies in 10
        days and would wait for idle_do_nothing (the replay, 2026-10-17)."""
        if len(normal) < _MIN_WAIT:
            return False
        return any(form != normal and "_" in form and form.startswith(normal) for form in self._forms)


def _called_name(markup: str) -> str | None:
    for pattern in _CALLED_NAME:
        found = pattern.search(markup)
        if found:
            return found.group(1)
    return None


def _leak(tool: LeakTool, shape: str) -> Verdict:
    return Verdict(CUT_AND_EXECUTE if tool.leak_safe else CUT, kind=_CALL, tool=tool.name, shape=shape)


def leaked_tool_name(text: str, tools: ToolNames, final: bool) -> Verdict | None:
    """Rule 1: a reply that is only a registered tool's name."""
    shapes = _shapes(text, final)
    if isinstance(shapes, list):
        for kind, body in shapes:
            name = _called_name(body) if kind == _CALL else None
            tool = tools.find(_normal(name)) if name else None
            if tool is not None:
                return _leak(tool, "markup")
        return None
    word = text.strip().strip(_WORD_EDGES).strip()
    if not _PLAIN.fullmatch(word):
        return None
    normal = _normal(word)
    tool = tools.find(normal)
    if not final:
        named = tool is not None and "_" in normal
        return Verdict(UNDECIDED) if named or tools.could_become(normal) else None
    if tool is None:
        return None
    if "_" in word:
        return _leak(tool, "underscore name")
    if "_" in normal:
        return _leak(tool, "spoken name")
    return Verdict(PLAY_AND_LOG, tool=tool.name, shape="plain word")


def artifact_shape(text: str, tools: ToolNames, final: bool) -> Verdict | None:
    """Rule 2: a reply made only of stage directions, call markup or thinking."""
    kind = spoken_artifact(text, final)
    if kind is None:
        return None
    if kind == _UNDECIDED:
        return None if final else Verdict(UNDECIDED)
    return Verdict(CUT, kind=kind)


Rule = Callable[[str, ToolNames, bool], Verdict | None]
RULES: Final[tuple[Rule, ...]] = (leaked_tool_name, artifact_shape)


def judge(text: str, tools: ToolNames, *, final: bool = False) -> Verdict:
    """The first decisive rule's verdict; UNDECIDED while any rule waits; else PLAY. Pure."""
    waiting = False
    for rule in RULES:
        verdict = rule(text, tools, final)
        if verdict is None:
            continue
        if verdict.action == UNDECIDED:
            waiting = True
            continue
        return verdict
    return Verdict(UNDECIDED) if waiting and not final else Verdict(PLAY)
