"""The output gate: what of a reply's transcript is spoken (2026-10-16, 2026-10-17).

One place decides whether a reply plays. Its rules run in order over the transcript so
far, each returning a typed Verdict or None (no opinion), and the first decisive one
wins:

1. leaked_tool_name: the reply is only a registered tool's name, bare or in call markup.
2. artifact_shape: the reply is only stage directions, call markup or thinking.
3. reasoning_header: the reply opens as reasoning written as plain text (11-08).

The handler holds audio while the verdict is undecided, cuts on CUT, and at the end of
the transcript (final=True) asks once more, so a bare name is judged on the whole reply.
The supervisor keeps no rule of its own: the transcript notification carries the cut's
kind, and a marked reply is never stored.
"""

from __future__ import annotations
import re
import json
from typing import Any, Final, Callable, Iterable
from dataclasses import replace, dataclass


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
_TAG_WAIT: Final[str] = (
    "finished tag"  # a finished tag ends the text so far: words may follow
)
_DIRECTION_MAX = 80  # characters inside a stage direction's brackets
_UNDECIDED_MAX = 200  # past this much text, an unfinished shape is a reply
_JSON_UNDECIDED_MAX = (
    2000  # an open JSON call, its first key already a call's: its query can be long
)
_GEMMA_QUOTE: Final[str] = '<|"|>'
_DIRECTION_CLOSERS: Final[dict[str, str]] = {"(": ")", "[": "]", "*": "*", "_": "_"}
_PAIRS: Final[dict[str, str]] = {"(": ")", "[": "]", "{": "}"}
# Whole special tokens and stray closing tags: <|"|>, <|im_end|>, <|tool_call>, <tool_call|>.
_TOKEN = re.compile(
    r"<\|[^<>|\s]{1,40}\|>|<\|\w{1,40}>|<\w{1,40}\|>"
    r"|</(?:tool_call|tool_code|function_call|function|think|thinking|call)\s*>",
    re.IGNORECASE,
)
# Wrappers whose whole body is markup: (opener, closer, kind).
_WRAPPERS = (
    (re.compile(r"<(think|thinking)>", re.IGNORECASE), "</{}>", _THINKING),
    (re.compile(r"<(tool_call|tool_code|function_call)>", re.IGNORECASE), "</{}>", _CALL),
    (re.compile(r"<function=[\w.-]+>", re.IGNORECASE), "</function>", _CALL),
)
_TAG_CALL = re.compile(r"<call:\s*[\w.-]+\s*", re.IGNORECASE)
_TAG_CALL_END = re.compile(r"[^<>]*?(/?)>")
_TAG_CALL_CLOSER = re.compile(r"</call(?::[\w.-]*)?\s*>", re.IGNORECASE)
# A registered tool's name written as a bare tag (11-07: "<stay_quiet/>" 17 times in one
# morning, voiced as a chuckle): <name/>, <name>, </name>, with whitespace and attributes.
_NAME_TAG = re.compile(r"<\s*/?\s*([A-Za-z_][\w.-]*)(?:\s[^<>]{0,200}?)?\s*/?\s*>")
_NAME_TAG_OPEN = re.compile(r"<\s*/?\s*([A-Za-z_][\w.-]*)[^<>]{0,200}")
# A tag's leading words, before any attribute's "=": "<stay quiet/>" names stay_quiet.
_TAG_WORDS = re.compile(r"<\s*/?\s*([A-Za-z_][\w.-]*(?:\s+[A-Za-z_][\w.-]*){0,5})")


def _tag_tool(tag: str, tools: ToolNames | None) -> LeakTool | None:
    """The tool a bare tag names: its longest run of leading words that is a tool's name."""
    words = _TAG_WORDS.match(tag)
    if tools is None or not words:
        return None
    parts = words.group(1).split()
    for n in range(len(parts), 0, -1):
        tool = tools.find(_normal(" ".join(parts[:n])))
        if tool is not None:
            return tool
    return None


def _skip_ssml(text: str, i: int, tools: ToolNames | None) -> int:
    """Return the index past whitespace, stray ". ! , ;" and SSML-like tags that name no
    tool: "<speak><stay_quiet/></speak>" is the tag alone (11-07, the review)."""
    while True:
        i = _skip(text, i)
        ssml = _SSML.match(text, i)
        if not ssml or _tag_tool(ssml.group(), tools) is not None:
            return i
        i = ssml.end()


_BARE_CALL = re.compile(r"call:\s*[\w.-]+\s*", re.IGNORECASE)
_BARE_CALL_OPEN = re.compile(r"call:\s*[\w.-]*\s*", re.IGNORECASE)
# A name with its arguments and no "call:" at all: "idle_do_nothing{reason:...}" (2026-10-17).
_NAMED_ARGS = re.compile(r"[A-Za-z_][\w.-]*(?=\{)")
# A registered tool written as a Python call anywhere in a reply (11-11: the next shape is caught
# the day it appears, not after 22 replies): "express(feeling=...)". Only registered names.
_PY_CALL = re.compile(r"[A-Za-z_][\w.]*(?=\()")
_CALL_ITEM = r"[A-Za-z_][\w.]*\(.*\)"
_CALL_LIST = re.compile(rf"\[\s*{_CALL_ITEM}(?:\s*,\s*{_CALL_ITEM})*\s*\]", re.DOTALL)
# A fenced block, ```tool_code (Gemma 3) or ```json: its tag, its body, and its closing
# fence, empty while it is still open.
_FENCE = re.compile(r"```\s*(tool_code|json)\b(.*?)(```|\Z)", re.IGNORECASE | re.DOTALL)
# A tool call spoken as JSON (2026-10-23 11:13:21, a pretty-printed recall call): a
# {"tool_calls": [...]} list, or one call object, {"function"|"name": ..., "arguments"|
# "parameters": ...}. The function may be an object with a name (and its arguments). The
# review of 9ec8ab9 added a bare [...] list, a ```json fence and null arguments; a call with
# no arguments key counts inside {"tool_calls": ...}, and elsewhere only when it names a
# known tool ({"name": "Matt"} is speech). While open it waits only if its first key could
# be a call's and is followed by ":"; anything else is released at once.
_JSON_CALL_KEYS: Final[frozenset[str]] = frozenset(
    {"name", "function", "arguments", "parameters", "type", "id"}
)
_JSON_FIRST_KEYS: Final[frozenset[str]] = _JSON_CALL_KEYS | {"tool_calls"}
_JSON_OPEN = re.compile(r'\{\s*(?:"([^"\\]*)("?))?')


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


def _json_arguments(value: object) -> bool:
    """Return whether `value` is arguments: an object, a JSON string of one, or null."""
    if value is None:
        return True
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return False
    return isinstance(value, dict)


def _calls(value: Any) -> list[dict[str, object]]:
    """Return decoded JSON's calls, each {"name": ..., "arguments": ...}; [] when it is not only calls.

    The JSON is a {"tool_calls": [...]} object, a bare list of calls or one call, whose keys
    are all call keys. A "function" object's name and arguments count as the call's own, the
    call's first; inside {"tool_calls": ...} a call with no arguments has null ones.
    """
    wrapped = isinstance(value, dict) and set(value) == {"tool_calls"}
    items = (
        value["tool_calls"]
        if wrapped
        else value if isinstance(value, list) else [value]
    )
    if not isinstance(items, list):
        return []
    calls: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict) or not set(item) <= _JSON_CALL_KEYS:
            return []
        function = item.get("function", item.get("name"))
        holders = [item]
        if isinstance(function, dict):
            if not set(function) <= _JSON_CALL_KEYS:
                return []
            holders.append(function)
            function = function.get("name")
        call: dict[str, object] = {"name": function}
        arguments = [
            holder[key]
            for holder in holders
            for key in ("arguments", "parameters")
            if key in holder
        ]
        if arguments or wrapped:
            call["arguments"] = arguments[0] if arguments else None
        calls.append(call)
    return calls


def _json_call_name(call: dict[str, object], tools: ToolNames | None) -> str | None:
    """Return a call's tool name, or None when it is not only a call.

    With no arguments it must name a known tool ({"name": "Matt"} is speech).
    """
    name = call["name"]
    if not isinstance(name, str) or not name:
        return None
    if "arguments" in call:
        return name if _json_arguments(call["arguments"]) else None
    return name if tools is not None and tools.find(_normal(name)) is not None else None


def _json_call_names(body: str, tools: ToolNames | None = None) -> list[str] | None:
    """Return the tool names of JSON text that is only tool calls, in order; else None.

    The one place a JSON block is decoded: a fence is stripped before it gets here.
    """
    try:
        value = json.loads(body)
    except ValueError:
        return None
    names = [_json_call_name(call, tools) for call in _calls(value)]
    return names if names and all(names) else None  # type: ignore[return-value]


def _could_be_json_call(rest: str) -> bool:
    """Return whether an open "{..." could still become a JSON call: its first key decides."""
    found = _JSON_OPEN.match(rest)
    if found is None:
        return False
    after = rest[found.end() :]
    key, closed = found.group(1), found.group(2)
    if key is None:
        return after == ""  # "{" and whitespace so far
    if not closed:
        return after == "" and any(first.startswith(key) for first in _JSON_FIRST_KEYS)
    return key in _JSON_FIRST_KEYS and after.lstrip()[:1] in ("", ":")


def _json_opening(rest: str) -> bool:
    """Return whether an open shape is a JSON call's start, which may wait past _UNDECIDED_MAX."""
    fence = _FENCE.match(rest)
    if rest.startswith("["):
        rest = rest[1:].lstrip()
    elif fence and fence.group(1).lower() == "json":
        rest = fence.group(2).lstrip()
    return _could_be_json_call(rest)


def _closed(kind: str, end: int) -> tuple[str, int]:
    """Return (kind, end) for a closed shape, or ("undecided", -1) when end is -1 (still open)."""
    return (kind, end) if end != -1 else (_UNDECIDED, -1)


def _tag_call_at(text: str, i: int, final: bool = False) -> tuple[str, int] | None:
    """Match a malformed Gemma 4 call at text[i]: <call:name/>, <call:name>...</call> and kin.

    A call whose arguments closed but whose tag never did ("<call:idle_do_nothing{reason:...}",
    2026-10-20) ends at its arguments: once words follow them, or at the end of the reply
    (`final`). Only "/>" or ">" right after the arguments close the tag, so a later ">" in
    the spoken words is never taken for it (the review)."""
    opener = _TAG_CALL.match(text, i)
    if not opener:
        return None
    j = opener.end()
    braced = j < len(text) and text[j] == "{"
    if braced:
        j = _balanced_end(text, j)
        if j == -1:
            return _UNDECIDED, -1
    if braced:
        after = text[j:].lstrip()
        if after.startswith(("/>", ">")):
            return _CALL, j + (len(text) - j - len(after)) + (
                2 if after[0] == "/" else 1
            )
        if after in ("", "/"):
            return (_CALL, j) if final else (_UNDECIDED, -1)
        return _CALL, j
    tag_end = _TAG_CALL_END.match(text, j)
    if not tag_end:
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


def _named_call(name: str, tools: ToolNames | None) -> bool:
    """A bare name{...} is a call when the name has an underscore or is a registered tool:
    "x{1}" in a reply is not (the review)."""
    return "_" in name or (tools is not None and tools.find(_normal(name)) is not None)


def _shape_at(
    text: str, i: int, final: bool = False, tools: ToolNames | None = None
) -> tuple[str, int, list[str] | None] | None:
    """Match the artifact at text[i]: (kind, end, names), ("undecided", -1, None) while open, or None.

    `names` are a JSON call's tools, decoded once here; None for any other shape.
    """
    rest = text[i:]
    c = text[i]
    if c == "<":
        angle = _angle_at(text, i, final)
        tagged = angle is not None and bool(_TAG_CALL.match(text, i))
        if angle is None and tools is not None:
            tag = _NAME_TAG.match(text, i)
            if tag and _tag_tool(tag.group(), tools) is not None:
                angle, tagged = (_CALL, tag.end()), True
        if angle is None:
            return None
        if (
            tagged
            and angle[0] == _CALL
            and not final
            and _skip_ssml(text, angle[1], tools) >= len(text)
        ):
            return _TAG_WAIT, -1, None  # a finished tag: words may follow it
        return (*angle, None)
    if rest[:5].lower() == "call:":
        call = _BARE_CALL.match(text, i)
        if call and call.end() < len(text) and text[call.end()] == "{":
            return (*_closed(_CALL, _balanced_end(text, call.end())), None)
        return (_UNDECIDED, -1, None) if _BARE_CALL_OPEN.fullmatch(rest) else None
    if "call:".startswith(rest.lower()):
        return _UNDECIDED, -1, None
    if c == "`":
        if not rest.startswith("```"):
            return (_UNDECIDED, -1, None) if "```".startswith(rest) else None
        fence = _FENCE.match(text, i)
        if fence and fence.group(3):
            if fence.group(1).lower() == "tool_code":
                return _CALL, fence.end(), None
            names = _json_call_names(fence.group(2), tools)
            return (_CALL, fence.end(), names) if names else None
        return (_UNDECIDED, -1, None) if rest.find("```", 3) == -1 else None
    if c == "{":
        end = _balanced_end(text, i)
        if end == -1:
            return (_UNDECIDED, -1, None) if _could_be_json_call(rest) else None
        names = _json_call_names(text[i:end], tools)
        return (_CALL, end, names) if names else None
    if c == "[":
        end = _balanced_end(text, i)
        if end != -1:
            if _CALL_LIST.fullmatch(text, i, end):
                return _CALL, end, None
            names = _json_call_names(text[i:end], tools)
            if names:
                return _CALL, end, names
        inner = text[i + 1 :].lstrip()
        if end == -1 and inner.startswith(
            "{"
        ):  # a bare list of JSON calls: an inner "]" is not its end
            return (_UNDECIDED, -1, None) if _could_be_json_call(inner) else None
    named = _NAMED_ARGS.match(text, i)
    if named and _named_call(named.group(), tools):
        return (*_closed(_CALL, _balanced_end(text, named.end())), None)
    py = _PY_CALL.match(text, i)
    if py and tools is not None and tools.find(_normal(py.group())) is not None:
        end = _balanced_end(text, py.end())
        if end == -1:
            return _UNDECIDED, -1, None
        inner = text[py.end() + 1 : end - 1]
        if not inner.strip() or "=" in inner or '"' in inner:  # "dance(s)" is a word
            return _CALL, end, None
    if c in _DIRECTION_CLOSERS:
        close_at = text.find(_DIRECTION_CLOSERS[c], i + 1)
        if close_at == -1:
            return _UNDECIDED, -1, None
        inner = text[i + 1 : close_at]
        if (
            inner.strip()
            and len(inner) <= _DIRECTION_MAX
            and not set(inner) & set("()[]*{}")
        ):
            return _STAGE, close_at + 1, None
    return None


def _shapes(
    text: str, final: bool = False, tools: ToolNames | None = None
) -> list[tuple[str, str, list[str] | None]] | str | None:
    """Return the artifact shapes `text` is made of, as (kind, text, JSON call names).

    The text is the shape's own; the names are a JSON call's tools, else None. "undecided"
    while a shape is still open; None when there are real words outside them.
    """
    head = text.strip()
    found: list[tuple[str, str, list[str] | None]] = []
    i = _skip_ssml(head, 0, tools)
    while i < len(head):
        shape = _shape_at(head, i, final, tools)
        if shape is None:
            return None
        kind, end, names = shape
        if kind == _TAG_WAIT:
            return _TAG_WAIT  # bounded by the spoken-name hold, not _UNDECIDED_MAX
        if kind == _UNDECIDED:
            limit = _JSON_UNDECIDED_MAX if _json_opening(head[i:]) else _UNDECIDED_MAX
            return _UNDECIDED if len(head) <= limit else None
        found.append((kind, head[i:end], names))
        i = _skip_ssml(head, end, tools)
    return found if found else _UNDECIDED


def spoken_artifact(text: str, final: bool = False) -> str | None:
    """Return "stage direction", "leaked tool call", "thinking", None (a reply), or "undecided".

    Text made only of artifact shapes, whitespace and stray ". ! , ;" is an artifact,
    named by its first shape. Pure.
    """
    shapes = _shapes(text, final)
    if shapes == _TAG_WAIT:
        return _UNDECIDED
    return shapes[0][0] if isinstance(shapes, list) else shapes


# SSML-like markup the model writes around words (11-03 19:31: "<speak>I'd love to!</speak>",
# copied reply to reply). The backend's voice reads it aloud, so it cannot be cut from audio
# already spoken: it is counted, kept out of the stored transcript, and the model told once.
_SSML = (
    re.compile(  # a tag as written: no space after "<", attributes only as name="value"
        r"</?(?:speak|prosody|break|emphasis|say-as|voice|phoneme|sub|audio|lang|mark)"
        r"""(?:\s+[\w:-]+\s*=\s*(?:"[^"<>]{0,200}"|'[^'<>]{0,200}'))*\s*/?>""",
        re.IGNORECASE,
    )
)


def without_ssml(text: str) -> tuple[str, int]:
    """`text` with SSML-like tags removed (their words kept), and how many there were."""
    found = _SSML.findall(text)
    if not found:
        return text, 0
    return re.sub(r"[ \t]{2,}", " ", _SSML.sub("", text)).strip(), len(found)


def without_calls(text: str, tools: ToolNames | None = None) -> str:
    """A played reply's text with its call markup taken out: what the supervisor keeps.

    The audio is untouched (a reply with real words plays whole, the owner, 2026-10-16),
    but kept markup reached his word-for-word memory and he copied it (2026-10-20). Pure.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "<" or i == 0 or not text[i - 1].isalnum():
            shape = _shape_at(text, i, final=True, tools=tools)
            if shape is not None and shape[0] in (_CALL, _THINKING):
                i = shape[1]
                continue
        out.append(text[i])
        i += 1
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()



_TAG_START = re.compile(r"<[A-Za-z_|/]")  # "a < b" is a comparison


def _words_in_brackets(inner: str) -> bool:
    return bool(inner.strip()) and "=" not in inner and '"' not in inner


def ends_in_open_call(text: str, tools: ToolNames | None = None) -> bool:
    """Whether `text` holds an unfinished call, which runs on to its end.

    "Sure. <tool_code> express(" when a barge-in cancels the reply mid-tag (the 11-11 review,
    F3): `without_calls` keeps such a fragment, since a closed shape is all it removes. A "<"
    starts one only right before a letter, "_", "|" or "/". An open bracket, star or
    underscore is a stage direction or words ("I like music (jazz"), and so is a word cut
    short that "call:" or a code fence starts with. Pure.
    """
    i = 0
    while i < len(text):
        if text[i] == "<" or i == 0 or not text[i - 1].isalnum():
            shape = _shape_at(text, i, final=True, tools=tools)
            if shape is not None and shape[0] == _UNDECIDED:
                rest = text[i:].lower()
                if text[i] in _DIRECTION_CLOSERS:
                    pass
                elif text[i] == "<" and not _TAG_START.match(text, i):
                    pass  # "I <3 you", "a < b": no tag starts there (the review of 99feb02)
                elif rest[:1] == "`" and not rest.startswith("```"):
                    pass
                elif rest[:1] == "c" and "call:".startswith(rest):
                    pass
                elif (py := _PY_CALL.match(text, i)) and _words_in_brackets(text[py.end() + 1 :]):
                    pass  # "dance(s" is a word, as `_shape_at` judges a closed one
                else:
                    return True
            elif shape is not None and shape[1] > i:
                i = shape[1]
                continue
        i += 1
    return False

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
    leaked ("markup", "json", "underscore name", "spoken name", "plain word"). `calls`
    holds every leaked call as its own verdict, this one's first: a JSON call list leaks
    several. It is empty when no tool leaked.
    """

    action: str
    kind: str | None = None
    tool: str | None = None
    shape: str | None = None
    calls: tuple[Verdict, ...] = ()
    waits_on_name: bool = False  # undecided on a name in words that would be cut


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
    re.compile(
        r"^([A-Za-z_][\w.-]*)\{"
    ),  # first: a name inside the arguments is not it (the review)
    re.compile(r"call:\s*([\w.-]+)", re.IGNORECASE),
    re.compile(r"<function=([\w.-]+)>", re.IGNORECASE),
    re.compile(r'"name"\s*:\s*"([^"]+)"'),
    re.compile(r"^\[\s*([A-Za-z_][\w.]*)\("),
    re.compile(r"^([A-Za-z_][\w.]*)\("),  # a bare Python call (11-11)
    re.compile(r"tool_code>?\s*([A-Za-z_][\w.]*)\(", re.IGNORECASE),  # 11-11: <tool_code> tag
    _NAME_TAG,
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
        """Whether more characters could make this a tool's name ("stay" -> "stay quiet",
        "expr" -> "express": a one-word name was released before its "{", 2026-10-20 22:59).

        Under _MIN_WAIT characters it never waits: "I" starts 133 of 659 replies in 10
        days and would wait for idle_do_nothing (the replay, 2026-10-17)."""
        if len(normal) < _MIN_WAIT:
            return False
        return any(form != normal and form.startswith(normal) for form in self._forms)

    def could_be_tag(self, name: str) -> bool:
        """Whether an open tag's name, lowercased, starts a tool's name."""
        return any(form.startswith(name) for form in self._forms)

    def could_be_cut(self, normal: str) -> bool:
        """Whether this is, or could become, a name that is cut when said alone: one with
        an underscore ("stay quiet", "stay"). A one-word name plays, so it never qualifies.
        """
        if len(normal) < _MIN_WAIT:
            return False
        return any("_" in form and form.startswith(normal) for form in self._forms)


def _called_name(markup: str) -> str | None:
    for pattern in _CALLED_NAME:
        found = pattern.search(markup)
        if found:
            return found.group(1)
    return None


def _leak(tool: LeakTool, shape: str) -> Verdict:
    return Verdict(
        CUT_AND_EXECUTE if tool.leak_safe else CUT,
        kind=_CALL,
        tool=tool.name,
        shape=shape,
    )


def _leaked(*calls: Verdict) -> Verdict:
    """Return the verdict on a reply of leaked calls: the first call's, with every call in `calls`."""
    return replace(calls[0], calls=calls)


def _ends_in_tool_tag(text: str, tools: ToolNames) -> bool:
    """Whether the text ends in a tool's bare tag, finished or still open ("<stay_qu").

    A finished one is judged on the whole reply, as words may follow it (the review), so
    its audio waits the spoken-name hold."""
    tail = text.rstrip(" \t\r\n.!,;")
    i = tail.rfind("<")
    while i != -1 and _SSML.fullmatch(tail, i) and _tag_tool(tail[i:], tools) is None:
        tail = tail[:i].rstrip(" \t\r\n.!,;")  # a closing </speak> after the tag
        i = tail.rfind("<")
    if i == -1:
        return False
    last = tail[i:]
    if last.lower().startswith("<call:") or "<call:".startswith(last.lower()):
        return True  # a tag call, finished or open: cut when nothing follows
    if _NAME_TAG.fullmatch(last):
        return _tag_tool(last, tools) is not None
    opened = _NAME_TAG_OPEN.fullmatch(last)
    return bool(opened) and tools.could_be_tag(opened.group(1).lower())


def leaked_tool_name(text: str, tools: ToolNames, final: bool) -> Verdict | None:
    """Rule 1: a reply that is only a registered tool's name."""
    shapes = _shapes(text, final, tools)
    if shapes == _TAG_WAIT:
        return Verdict(UNDECIDED, waits_on_name=True)
    if shapes == _UNDECIDED and not final:
        return Verdict(
            UNDECIDED, waits_on_name=_ends_in_tool_tag(text, tools)
        )  # a registered name's open arguments: "express{feel"
    if isinstance(shapes, list):
        for kind, body, json_names in shapes:
            if kind != _CALL:
                continue
            names = json_names if json_names is not None else [_called_name(body)]
            shape = "json" if json_names is not None else "markup"
            found = [tools.find(_normal(name)) for name in names if name]
            if json_names is None and _NAME_TAG.fullmatch(body):
                found = [_tag_tool(body, tools)]  # "<stay quiet/>": its words
            leaks = [_leak(tool, shape) for tool in found if tool is not None]
            if leaks:
                return _leaked(*leaks)
        return None
    word = without_ssml(text)[0].strip().strip(_WORD_EDGES).strip()
    if not _PLAIN.fullmatch(word):
        return None
    normal = _normal(word)
    tool = tools.find(normal)
    if not final:
        # A tool's name, or the start of one, waits for what follows: "{" makes it a
        # call, a space and a word make it a sentence (the owner, 2026-10-20).
        named = tool is not None and len(normal) >= _MIN_WAIT
        if not (named or tools.could_become(normal)):
            return None
        return Verdict(UNDECIDED, waits_on_name=tools.could_be_cut(normal))
    if tool is None:
        return None
    if "_" in word:
        return _leaked(_leak(tool, "underscore name"))
    if "_" in normal:
        return _leaked(_leak(tool, "spoken name"))
    return _leaked(Verdict(PLAY_AND_LOG, tool=tool.name, shape="plain word"))


def artifact_shape(text: str, tools: ToolNames, final: bool) -> Verdict | None:
    """Rule 2: a reply made only of stage directions, call markup or thinking."""
    kind = spoken_artifact(text, final)
    if kind is None:
        return None
    if kind == _UNDECIDED:
        return None if final else Verdict(UNDECIDED)
    return Verdict(CUT, kind=kind)


# Thinking written as plain text (11-08 08:37:56): a follow-up after express opened
# "<thought\nThinking Process:\n1. Current state: ..." with no <think> tags, and all 6,000
# characters of it were spoken. Two shapes have evidence, and only they are cut whole: Gemma
# 4's thought marker at the start, followed by anything, and a "Thinking Process" or
# "Thought Process" header (bold or a heading), then ":" or a newline. Losing a real answer
# is worse, so nothing else is: "Plan:", "Analysis:" and "1. Leave at 7:30" play. The
# opening is held only while it is a prefix of one of the shapes, from 4 characters on.
_THOUGHT_MARKS: Final[tuple[str, ...]] = ("<|channel>thought", "<thought")
_REASONING_HEADERS: Final[tuple[str, ...]] = ("thinking process", "thought process")
_DECORATION = re.compile(r"(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*")
_HEADER_END = re.compile(r"[ \t]*(?:\*\*|__)?[ \t]*(?::|\n)")
_HEADER_END_OPEN = re.compile(r"[ \t]*(?:\*\*?|__?)?[ \t]*")
_REASONING_MIN_WAIT = 4  # "In", "An", "1." and "#" never wait


def reasoning_header(text: str, tools: ToolNames, final: bool) -> Verdict | None:
    """Rule 3: a reply that opens as reasoning, its thinking written as plain text."""
    head = text.lstrip()
    lower = head.lower()
    if lower.startswith(_THOUGHT_MARKS):
        return Verdict(CUT, kind=_THINKING)
    rest = _DECORATION.sub("", head, count=1)  # it always matches at 0, if only empty
    words = rest.lower()
    waits = any(mark.startswith(lower) for mark in _THOUGHT_MARKS)
    for header in _REASONING_HEADERS:
        if words.startswith(header):
            after = rest[len(header) :]
            if _HEADER_END.match(after):
                return Verdict(CUT, kind=_THINKING)
            waits = waits or _HEADER_END_OPEN.fullmatch(after) is not None
        waits = waits or header.startswith(words)
    if waits and not final and len(head) >= _REASONING_MIN_WAIT:
        return Verdict(UNDECIDED)
    return None


Rule = Callable[[str, ToolNames, bool], Verdict | None]
RULES: Final[tuple[Rule, ...]] = (leaked_tool_name, artifact_shape, reasoning_header)


def judge(text: str, tools: ToolNames, *, final: bool = False) -> Verdict:
    """The first decisive rule's verdict; UNDECIDED while any rule waits; else PLAY. Pure."""
    waiting: Verdict | None = None
    for rule in RULES:
        verdict = rule(text, tools, final)
        if verdict is None:
            continue
        if verdict.action == UNDECIDED:
            waiting = waiting or verdict
            continue
        return verdict
    return waiting if waiting and not final else Verdict(PLAY)


# The leaked feeling (11-11): "express(feeling=..., intensity=...)" written as text never ran, so
# his face stopped showing what he felt (18 of 22 leaks one morning). Its arguments are read back
# so the call can run; express checks them itself.
_PY_EXPRESS = re.compile(r"\bexpress\s*\(([^()]*)\)")
_GEMMA_EXPRESS = re.compile(r"call:\s*express\s*\{([^{}]*)\}", re.IGNORECASE)
_GEMMA_ARG = re.compile(r'(\w+)\s*:\s*(?:<\|"\|>(.*?)<\|"\|>|"([^"]*)"|([\w.+-]+))')


def leaked_express(text: str) -> dict[str, Any] | None:
    """The arguments of the first express call written in `text` as a Python call or Gemma's
    call markup, as literals; None without one. Pure."""
    import ast

    m = _PY_EXPRESS.search(text)
    if m:
        try:
            call = ast.parse(f"f({m.group(1)})", mode="eval").body
        except SyntaxError:
            return None
        if not isinstance(call, ast.Call) or call.args:
            return None
        out: dict[str, Any] = {}
        for kw in call.keywords:
            try:
                out[str(kw.arg)] = ast.literal_eval(kw.value)
            except ValueError:
                return None
        return out or None
    m = _GEMMA_EXPRESS.search(text)
    if m:
        out = {}
        for key, q, d, bare in _GEMMA_ARG.findall(m.group(1)):
            value: Any = q or d or bare
            if bare:
                try:
                    value = float(bare)
                except ValueError:
                    pass
            out[key] = value
        return out or None
    return None
