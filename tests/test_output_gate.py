"""The output gate's tool-name rule (the planning session, 2026-10-17).

The cases are generated from tool registries: every tool times every shape it could
leak in. The artifact rule's own table is SPOKEN_ARTIFACT_CASES in
test_huggingface_realtime.py.
"""

from __future__ import annotations

from typing import Any

import pytest

from reachy_mini_conversation_app import output_gate as gate
from reachy_mini_conversation_app.tools import core_tools
from reachy_mini_conversation_app.tools.idle_do_nothing import IdleDoNothing


# The companion's live registry, from the app's "Tools to be used in conversation" line
# at 20:41:44 on 2026-10-17.
COMPANION_TOOLS = (
    "dance stop_dance play_emotion stop_emotion idle_do_nothing move_head sweep_look "
    "head_tracking camera task_status task_cancel boats set_commitment list_commitments "
    "cancel_commitment express flights look skills_list skill_view transit events why_lit "
    "remember forget turn_body look_at_me music stay_quiet stop_listening web_search "
    "pollen_robotics_reachy_mini_time_tool__get_time "
    "pollen_robotics_reachy_mini_weather_tool__get_weather"
).split()
LEAK_SAFE = {"stay_quiet", "idle_do_nothing"}


def _registry() -> list[str]:
    names = set(COMPANION_TOOLS)
    names.update(spec["name"] for spec in core_tools.get_tool_specs())
    return sorted(names)


REGISTRY = _registry()
TOOLS = gate.ToolNames((name, name in LEAK_SAFE) for name in REGISTRY)


def _spoken(name: str) -> str:
    """The name as words: its part after "__", with spaces for underscores."""
    return name.rsplit("__", 1)[-1].replace("_", " ")


MARKUP_SHAPES = (
    "call:{n}{{}}",
    " CALL: {n} {{}} ",
    "<call:{n}/>",
    '<call:{n} madad="true"/>',
    '<call:{n}{{reason:<|"|>Indistinct noise, not addressed to me.<|"|>}}/>',
    "<|tool_call>call:{n}{{}}<tool_call|>",
    '<tool_call>{{"name": "{n}", "arguments": {{}}}}</tool_call>',
    "<function={n}><parameter=x>1</parameter></function>",
    "[{n}()]",
    "```tool_code\n{n}()\n```",
    '{n}(reason="ok")',  # 11-11: a bare Python call, any tag or none
    "<tool_code> {n}() </tool_code>",  # 11-11: Gemma's tag, 22 replies in half an hour
    '<tool_code>{n}(feeling="happy", intensity=0.6)</tool_code>',
    "{n}{{}}",  # bare, no "call:" (2026-10-17, 2026-10-18)
    '{n}{{reason:<|"|>Conversational acknowledgement.<|"|>}}',
)


def _expected_cut(name: str) -> str:
    return gate.CUT_AND_EXECUTE if name in LEAK_SAFE else gate.CUT


@pytest.mark.parametrize("name", REGISTRY)
@pytest.mark.parametrize("shape", MARKUP_SHAPES)
def test_every_tool_in_every_markup_shape_is_cut(name: str, shape: str) -> None:
    verdict = gate.judge(shape.format(n=name), TOOLS, final=True)
    assert (verdict.action, verdict.tool, verdict.shape) == (
        _expected_cut(name),
        name,
        "markup",
    )
    assert verdict.kind == "leaked tool call"


@pytest.mark.parametrize("name", REGISTRY)
@pytest.mark.parametrize("shape", ("{n}", "{n}.", " {n}! ", "`{n}`", '"{n}"'))
def test_every_tool_said_bare_is_judged_on_the_whole_reply(
    name: str, shape: str
) -> None:
    """An underscore name is cut; a one-word name plays, logged; mid-reply, either waits
    or plays, never cuts.
    """
    text = shape.format(n=name)
    verdict = gate.judge(text, TOOLS, final=True)
    if "_" in name:
        assert (verdict.action, verdict.tool, verdict.shape) == (
            _expected_cut(name),
            name,
            "underscore name",
        )
        assert gate.judge(text, TOOLS).action == gate.UNDECIDED
    else:
        assert (verdict.action, verdict.tool, verdict.shape) == (
            gate.PLAY_AND_LOG,
            name,
            "plain word",
        )
        assert gate.judge(text, TOOLS).action in (gate.PLAY, gate.UNDECIDED)


@pytest.mark.parametrize(
    "name", [n for n in REGISTRY if "_" in _spoken(n).strip() or " " in _spoken(n)]
)
@pytest.mark.parametrize("shape", ("{w}", "{W}.", "{w}!", "{h}"))
def test_every_multi_word_tool_said_as_words_is_cut(name: str, shape: str) -> None:
    words = _spoken(name)
    text = shape.format(w=words, W=words.capitalize(), h=words.replace(" ", "-"))
    verdict = gate.judge(text, TOOLS, final=True)
    assert (verdict.action, verdict.tool, verdict.shape) == (
        _expected_cut(name),
        name,
        "spoken name",
    )
    assert gate.judge(text, TOOLS).action == gate.UNDECIDED  # more words may follow


@pytest.mark.parametrize(
    "text",
    [
        "Yes, that's true. I try to stay quiet so I don't interrupt your conversation.",
        "I'll stay quiet (for now).",
        "Stay warm tonight.",
        "Do you want me to dance?",
        "Let's look at the boats.",
        "The music was lovely.",
        "I can do a web search for that.",
        "Look at me go!",
        "Stop listening to him, he's teasing.",
        "Sure!",
        "Richie here.",
        "call:stay_quiet{} and then hello",
        '{"status": "ok", "server_alias": "pollen_robotics_reachy_mini_weather_tool"}',
    ],
)
def test_natural_sentences_play(text: str) -> None:
    assert gate.judge(text, TOOLS, final=True) == gate.Verdict(gate.PLAY)


@pytest.mark.parametrize("text", ["Dance!", "Music.", "look", "Remember?", "Events"])
def test_a_single_plain_word_tool_plays_and_is_logged(text: str) -> None:
    verdict = gate.judge(text, TOOLS, final=True)
    assert verdict.action == gate.PLAY_AND_LOG
    assert verdict.tool == text.strip(".!?").lower()


@pytest.mark.parametrize(
    "text", ["Sta", "stay", "stay ", "stay_", "stay qu", "look at", "web", "Why"]
)
def test_the_start_of_a_multi_word_name_waits(text: str) -> None:
    assert gate.judge(text, TOOLS).action == gate.UNDECIDED
    assert gate.judge(text, TOOLS, final=True).action in (gate.PLAY, gate.PLAY_AND_LOG)


@pytest.mark.parametrize("text", ["I", "I'm", "He", "We", "St", "Go"])
def test_a_reply_never_waits_on_a_short_start(text: str) -> None:
    """ "I" starts a fifth of his replies; it would otherwise wait for idle_do_nothing."""
    assert gate.judge(text, TOOLS).action == gate.PLAY


def test_an_unregistered_call_is_still_an_artifact_with_no_tool() -> None:
    verdict = gate.judge("call:teleport{}", TOOLS, final=True)
    assert verdict == gate.Verdict(gate.CUT, kind="leaked tool call")


def test_rules_run_in_order_from_one_declaration() -> None:
    assert gate.RULES == (
        gate.leaked_tool_name,
        gate.artifact_shape,
        gate.reasoning_header,
    )


def test_with_no_registry_the_artifact_rule_alone_decides() -> None:
    empty = gate.ToolNames()
    assert gate.judge("call:stay_quiet{}", empty, final=True) == gate.Verdict(
        gate.CUT, kind="leaked tool call"
    )
    assert gate.judge("stay_quiet", empty, final=True) == gate.Verdict(gate.PLAY)
    assert gate.judge("", empty) == gate.Verdict(gate.UNDECIDED)
    assert gate.judge("", empty, final=True) == gate.Verdict(gate.PLAY)


def test_tools_opt_in_to_running_when_leaked() -> None:
    assert core_tools.Tool.leak_safe is False
    assert IdleDoNothing.leak_safe is True


# 2026-10-20 evening: 4 of 19 replies leaked Gemma call markup. The whole-reply leaks had
# no closing "/>" and stayed undecided, so they were spoken and kept; the leading ones were
# spoken (real words follow) and kept markup and all, and his word-for-word memory taught
# him to do it again.
Q = '<|"|>'
OPEN_IDLE = f"<call:idle_do_nothing{{reason:{Q}Acknowledging Matt's comment.{Q}}}"
LEADING = (
    f"<call:express{{feeling:{Q}shy{Q},intensity:0.4}}/>I'm sorry I'm being glitchy."
)


def test_a_call_left_open_at_the_end_of_the_reply_is_cut() -> None:
    assert gate.judge(OPEN_IDLE, TOOLS, final=False).action == gate.UNDECIDED
    verdict = gate.judge(OPEN_IDLE, TOOLS, final=True)
    assert (verdict.action, verdict.tool, verdict.shape) == (
        gate.CUT_AND_EXECUTE,
        "idle_do_nothing",
        "markup",
    )


def test_an_open_call_before_words_ends_at_its_arguments() -> None:
    text = f"<call:express{{feeling:{Q}oops{Q},intensity:0.5}} I'm still here, Matt!"
    assert gate.judge(text, TOOLS, final=True).action == gate.PLAY
    assert gate.without_calls(text) == "I'm still here, Matt!"


@pytest.mark.parametrize(
    "text, kept",
    [
        (LEADING, "I'm sorry I'm being glitchy."),
        ("Sure. <call:look_at_me/> There you are.", "Sure. There you are."),
        ("call:express{} Hello!", "Hello!"),
        (f"express{{feeling:{Q}shy{Q}}} Am I?", "Am I?"),
        ("Use {curly} braces.", "Use {curly} braces."),
        ("I'll recall: the ferry left at 3.", "I'll recall: the ferry left at 3."),
        ("(Laughs) yes, it leaves at 3", "(Laughs) yes, it leaves at 3"),
        ("Plain reply.", "Plain reply."),
    ],
)
def test_a_played_reply_is_kept_without_its_call_markup(text: str, kept: str) -> None:
    assert gate.judge(text, TOOLS, final=True).action == gate.PLAY
    assert gate.without_calls(text, TOOLS) == kept


# The /code-review of 2026-10-20.


def test_the_name_before_the_arguments_is_the_leaked_tool() -> None:
    verdict = gate.judge('remember{"name":"Alli","fact":"x"}', TOOLS, final=True)
    assert (verdict.tool, verdict.action) == ("remember", gate.CUT)
    verdict = gate.judge('stay_quiet{reason:"call: later"}', TOOLS, final=True)
    assert (verdict.tool, verdict.action) == ("stay_quiet", gate.CUT_AND_EXECUTE)


@pytest.mark.parametrize(
    "text, kept",
    [
        (
            "Sure! <call:express{emotion:happy} That's great, 5 > 3 right?",
            "Sure! That's great, 5 > 3 right?",
        ),
        ("I set x{1} to two and y{2}", "I set x{1} to two and y{2}"),
        ('Okay {"name": "stay_quiet", "arguments": {}} fine', "Okay fine"),
        ("Okay ```tool_code\nstay_quiet()\n``` fine", "Okay fine"),
        ("Okay. <think>hmm</think> Hello", "Okay. Hello"),
    ],
)
def test_review_cases_keep_the_spoken_words_and_only_them(text: str, kept: str) -> None:
    assert gate.without_calls(text, TOOLS) == kept


def test_a_word_with_braces_that_is_no_tool_is_a_reply() -> None:
    assert gate.judge("Hi{there}", TOOLS, final=True).action == gate.PLAY


def test_an_open_tag_call_waits_while_streaming_and_ends_at_its_arguments_once_words_follow() -> (
    None
):
    head = f"<call:express{{feeling:{Q}shy{Q}}}"
    assert gate.spoken_artifact(head) == gate.UNDECIDED
    assert gate.spoken_artifact(head + " /") == gate.UNDECIDED
    assert gate.spoken_artifact(head + " />") == gate.UNDECIDED  # words may follow it
    assert gate.spoken_artifact(head + " />", final=True) == "leaked tool call"
    assert gate.spoken_artifact(head + " Hello") is None  # words: a reply


# 2026-10-20 22:59: "express{feeling:<|"|>delight<|"|>}" was heard. A one-word tool name was
# released before its "{" arrived; now the name, and the start of it, waits (the owner).
BARE_EXPRESS = f"express{{feeling:{Q}delight{Q}}}"


@pytest.mark.parametrize("cut_at", [3, 5, 7, 8, 12, len(BARE_EXPRESS) - 1])
def test_a_one_word_tool_name_waits_for_what_follows(cut_at: int) -> None:
    assert gate.judge(BARE_EXPRESS[:cut_at], TOOLS).action == gate.UNDECIDED


def test_the_2259_bare_express_is_cut_once_its_arguments_close() -> None:
    verdict = gate.judge(BARE_EXPRESS, TOOLS)
    assert (verdict.action, verdict.tool, verdict.shape) == (
        gate.CUT,
        "express",
        "markup",
    )
    assert gate.judge(BARE_EXPRESS, TOOLS, final=True).action == gate.CUT


@pytest.mark.parametrize(
    "text", ["Express yourself", "Expressive", "Music is nice", "Looking good"]
)
def test_a_sentence_that_starts_like_a_tool_name_plays_at_the_next_word(
    text: str,
) -> None:
    assert gate.judge(text, TOOLS).action == gate.PLAY


# 2026-10-23 11:13:21: a recall call spoken as JSON. A JSON call is its own shape, "json"
# (the {"name", "arguments"} object was listed with the markup shapes before).
JSON_SHAPES = (
    '{{"name": "{n}", "arguments": {{}}}}',
    '{{"tool_calls": [{{"function": "{n}", "parameters": {{"query": "how we met"}}}}]}}',
    '{{\n  "tool_calls": [\n    {{\n      "function": "{n}",\n      "parameters": {{}}\n    }}\n  ]\n}}',
    '{{"tool_calls": [{{"name": "{n}", "arguments": "{{\\"x\\": 1}}"}}]}}',
    '{{"tool_calls": [{{"type": "function", "function": {{"name": "{n}", "arguments": "{{}}"}}}}]}}',
    '{{"function": "{n}", "parameters": {{}}}}',
    '{{"function": {{"name": "{n}"}}, "arguments": {{}}}}',
    # the review of 9ec8ab9: no arguments key, null arguments, a ```json fence, a bare list
    '{{"function": "{n}"}}',
    '{{"name": "{n}"}}',
    '{{"tool_calls": [{{"function": "{n}"}}]}}',
    '{{"name": "{n}", "arguments": null}}',
    '{{"tool_calls": [{{"function": "{n}", "parameters": null}}]}}',
    '```json\n{{"function": "{n}", "parameters": {{"q": "x"}}}}\n```',
    '```json\n{{"tool_calls": [{{"name": "{n}", "arguments": {{}}}}]}}\n```',
    '[{{"function": "{n}", "parameters": {{"items": [1, 2]}}}}]',
    '[\n  {{"name": "{n}"}}\n]',
)


@pytest.mark.parametrize(
    "name, shape",
    [("dance", shape) for shape in JSON_SHAPES]
    + [(name, JSON_SHAPES[1]) for name in REGISTRY],
)
def test_every_json_shape_and_every_tool_is_cut(name: str, shape: str) -> None:
    """Every shape with one tool, and every tool in the 10-23 shape: the name is read the same in each."""
    verdict = gate.judge(shape.format(n=name), TOOLS, final=True)
    assert (verdict.action, verdict.tool, verdict.shape) == (
        _expected_cut(name),
        name,
        "json",
    )


def test_a_json_call_list_names_each_of_its_tools() -> None:
    text = '{"tool_calls": [{"function": "stay_quiet", "parameters": {}}, {"name": "dance", "arguments": {}}]}'
    verdict = gate.judge(text, TOOLS, final=True)
    assert (verdict.action, verdict.tool) == (gate.CUT_AND_EXECUTE, "stay_quiet")
    assert [(call.action, call.tool, call.shape) for call in verdict.calls] == [
        (gate.CUT_AND_EXECUTE, "stay_quiet", "json"),
        (gate.CUT, "dance", "json"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        '{"tool_calls": [{"function": "teleport", "parameters": {}}]}',
        '{"function": "teleport", "parameters": {}}',
    ],
)
def test_an_unknown_json_call_is_an_artifact_with_no_tool(text: str) -> None:
    assert gate.judge(text, TOOLS, final=True) == gate.Verdict(
        gate.CUT, kind="leaked tool call"
    )


JSON_CALL = '{\n  "tool_calls": [{"function": "recall", "parameters": {"query": "how we met"}}]}'


@pytest.mark.parametrize(
    "text",
    [
        JSON_CALL[:2],  # "{" and whitespace
        JSON_CALL[:9],  # inside the first key
        JSON_CALL[:60],  # inside the arguments
        '{"name"',
        '{"name" ',
        '{"name":',
        '{"name" :',
        "[",
        "[{",
        '[\n  {"func',
        "```js",
        '```json\n{"tool_ca',
    ],
)
def test_the_start_of_a_json_call_waits(text: str) -> None:
    assert gate.judge(text, TOOLS).action == gate.UNDECIDED


@pytest.mark.parametrize(
    "text",
    [
        "{a",
        "{ a, b",
        '{"query"',
        '{"quer"y',
        '{"tool_calls": 3}',
        '{"name": "x", "colour": "red"}',
        "{}",
    ],
)
def test_braces_that_cannot_be_a_call_are_released_at_once(text: str) -> None:
    assert gate.judge(text, TOOLS).action == gate.PLAY


def test_a_long_json_call_still_waits_past_the_markup_limit() -> None:
    text = (
        '{"tool_calls": [{"function": "recall", "parameters": {"query": "' + "a" * 300
    )
    tools = gate.ToolNames(
        [("recall", False)]
    )  # added to the companion's list after 10-17
    assert gate.judge(text, tools).action == gate.UNDECIDED
    verdict = gate.judge(text + '"}}]}', tools)
    assert (verdict.action, verdict.tool, verdict.shape) == (gate.CUT, "recall", "json")


@pytest.mark.parametrize(
    "text, kept",
    [
        (
            'Let me check. {"tool_calls": [{"function": "recall", "parameters": {}}]} Hmm.',
            "Let me check. Hmm.",
        ),
        ("the set {a, b}", "the set {a, b}"),
    ],
)
def test_a_played_reply_is_kept_without_its_json_call(text: str, kept: str) -> None:
    assert gate.judge(text, TOOLS, final=True).action == gate.PLAY
    assert gate.without_calls(text, TOOLS) == kept


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "Matt"}',
        '{"function": "teleport"}',
        '[{"name": "Matt"}]',
        '{"name": "Matt", "arguments": null, "age": 3}',
        '```json\n{"name": "Matt"}\n```',
    ],
)
def test_a_json_object_with_no_arguments_is_a_call_only_for_a_known_tool(
    text: str,
) -> None:
    assert gate.judge(text, TOOLS, final=True).action == gate.PLAY


def test_a_wrapped_call_with_no_arguments_is_cut_even_when_unknown() -> None:
    assert gate.judge(
        '{"tool_calls": [{"function": "teleport"}]}', TOOLS, final=True
    ) == gate.Verdict(gate.CUT, kind="leaked tool call")


@pytest.mark.parametrize(
    "text", ['{"name" is how you write it', '{"function"s are', '{"name" ,']
)
def test_a_closed_first_key_with_no_colon_is_released_at_once(text: str) -> None:
    assert gate.judge(text, TOOLS).action == gate.PLAY


def test_a_bare_list_with_a_nested_list_is_still_a_call() -> None:
    text = '[{"function": "remember", "parameters": {"facts": ["a", "b"]}}]'
    assert (
        gate.judge(text[:-1], TOOLS).action == gate.UNDECIDED
    )  # the inner "]" is not the end
    verdict = gate.judge(text, TOOLS)
    assert (verdict.action, verdict.tool, verdict.shape) == (
        gate.CUT,
        "remember",
        "json",
    )


def test_ssml_tags_are_counted_and_kept_out_of_the_stored_reply() -> None:
    """11-03 19:31: "<speak>I'd love to!</speak>", read aloud by the backend's voice."""
    from reachy_mini_conversation_app.output_gate import without_ssml

    assert without_ssml("<speak>I'd love to!</speak>") == ("I'd love to!", 2)
    for plain in ("if x < s and y > 3, done", "5 <s 7 >2", "a <voice of reason> here"):
        assert without_ssml(plain) == (
            plain,
            0,
        )  # the review (11-03): ordinary text kept
    assert without_ssml(
        'Wait <break time="1s"/> there <prosody rate="slow">it is</prosody>.'
    ) == ("Wait there it is.", 3)
    assert without_ssml("I'd say 3 < 5 and <think> is not ours") == (
        "I'd say 3 < 5 and <think> is not ours",
        0,
    )


# The bare tag (11-07): the model wrote its stay_quiet call as "<stay_quiet/>" 17 times in
# one morning; the gate let it through and the voice rendered it as a chuckle.
BARE_TAG_SHAPES = (
    "<{n}/>",
    "<{n}>",
    "</{n}>",
    "< {n} />",
    "<{N}/>",
    '<{n} reason="side talk"/>',
    "<{n}/>.",
)


@pytest.mark.parametrize("name", REGISTRY)
@pytest.mark.parametrize("shape", BARE_TAG_SHAPES)
def test_every_tool_as_a_bare_tag_is_cut(name: str, shape: str) -> None:
    """Mid-reply it waits on the long hold, as words may follow; the whole reply is cut."""
    text = shape.format(n=name, N=name.upper())
    verdict = gate.judge(text, TOOLS)
    assert (verdict.action, verdict.waits_on_name) == (gate.UNDECIDED, True)
    verdict = gate.judge(text, TOOLS, final=True)
    assert (verdict.action, verdict.tool, verdict.shape, verdict.kind) == (
        _expected_cut(name),
        name,
        "markup",
        "leaked tool call",
    )


@pytest.mark.parametrize("text", ["<sta", "<stay_quiet", "<stay_quiet/", "</sta"])
def test_the_start_of_a_bare_tag_waits_on_the_name(text: str) -> None:
    verdict = gate.judge(text, TOOLS)
    assert (verdict.action, verdict.waits_on_name) == (gate.UNDECIDED, True)


def test_a_bare_tag_with_words_after_it_in_the_same_text_plays() -> None:
    assert gate.judge("<stay_quiet/> Sure", TOOLS) == gate.Verdict(gate.PLAY)


@pytest.mark.parametrize(
    "text",
    [
        "<stay_quiet/> Sure, the train leaves at three.",
        "<b>Hello</b>",
        "<not_a_tool/>",
        "5 < 6 and stay > 2",
    ],
)
def test_a_bare_tag_with_real_words_or_no_tool_plays(text: str) -> None:
    assert gate.judge(text, TOOLS, final=True) == gate.Verdict(gate.PLAY)


def test_a_played_reply_is_kept_without_its_bare_tag() -> None:
    assert gate.without_calls("<stay_quiet/> Sure.", TOOLS) == "Sure."
    assert gate.without_calls("<b>Hello</b>", TOOLS) == "<b>Hello</b>"


@pytest.mark.parametrize(
    "text", ["Stay quiet.", "stay quiet", "Stay", "stay qu", "Idle do"]
)
def test_a_name_in_words_that_could_be_cut_waits_on_the_name(text: str) -> None:
    """Its audio is held until the verdict (SPOKEN_NAME_HOLD_S), not the short hold."""
    verdict = gate.judge(text, TOOLS)
    assert (verdict.action, verdict.waits_on_name) == (gate.UNDECIDED, True)


@pytest.mark.parametrize("text", ["Dance", "Expr", "<xyz", "(stops"])
def test_a_wait_that_cannot_end_in_a_spoken_name_cut_is_the_short_hold(
    text: str,
) -> None:
    """A one-word name ends as "play and log"; markup decides on its own deltas."""
    verdict = gate.judge(text, TOOLS)
    assert (verdict.action, verdict.waits_on_name) == (gate.UNDECIDED, False)


@pytest.mark.parametrize(
    "text", ["Sure.", "Stay calm, it's fine", "Yes.", "I'll stay quiet while you talk."]
)
def test_ordinary_short_replies_do_not_wait(text: str) -> None:
    assert gate.judge(text, TOOLS) == gate.Verdict(gate.PLAY)


# The three leaks the review of a67a150 found (the owner, 11-07).
MULTI_WORD = [n for n in REGISTRY if "_" in n.rsplit("__", 1)[-1]]


@pytest.mark.parametrize("name", MULTI_WORD)
@pytest.mark.parametrize(
    "shape", ("<{w}/>", "< {w} />", "<{W}>", '<{w} reason="side talk"/>', "</{w}>")
)
def test_a_tool_name_with_spaces_in_a_tag_is_cut(name: str, shape: str) -> None:
    words = _spoken(name)
    text = shape.format(w=words, W=words.upper())
    verdict = gate.judge(text, TOOLS)
    assert (verdict.action, verdict.waits_on_name) == (gate.UNDECIDED, True)
    verdict = gate.judge(text, TOOLS, final=True)
    assert (verdict.action, verdict.tool, verdict.shape) == (
        _expected_cut(name),
        name,
        "markup",
    )


@pytest.mark.parametrize(
    "text",
    [
        "<speak><stay_quiet/></speak>",
        "<speak> <stay_quiet/> </speak>",
        '<prosody rate="slow"><stay_quiet/></prosody>',
        "<speak><call:stay_quiet/></speak>",
        "<speak><stay quiet/></speak>",
    ],
)
def test_a_tool_tag_inside_ssml_is_cut(text: str) -> None:
    verdict = gate.judge(text, TOOLS)
    assert (verdict.action, verdict.waits_on_name) == (gate.UNDECIDED, True)
    verdict = gate.judge(text, TOOLS, final=True)
    assert (verdict.action, verdict.tool) == (gate.CUT_AND_EXECUTE, "stay_quiet")


@pytest.mark.parametrize(
    "text", ["<call:stay_quiet/>", "<call:stay_quiet>", "<call:express{}/>"]
)
def test_a_finished_tag_call_waits_for_what_follows(text: str) -> None:
    verdict = gate.judge(text, TOOLS)
    assert (verdict.action, verdict.waits_on_name) == (gate.UNDECIDED, True)
    assert gate.judge(text, TOOLS, final=True).action in (
        gate.CUT,
        gate.CUT_AND_EXECUTE,
    )


@pytest.mark.parametrize(
    "text",
    [
        "<stay calm/>",
        "<speak>Sure.</speak>",
        "<speak><stay_quiet/> Sure, it leaves at three.</speak>",
        "<call:stay_quiet/> Sure, it leaves at three.",
    ],
)
def test_tags_around_real_words_play(text: str) -> None:
    assert gate.judge(text, TOOLS) == gate.Verdict(gate.PLAY)
    assert gate.judge(text, TOOLS, final=True) == gate.Verdict(gate.PLAY)


def test_a_played_reply_is_kept_without_its_spaced_tag() -> None:
    assert gate.without_calls("<stay quiet/> Sure.", TOOLS) == "Sure."
    assert gate.without_calls("<stay calm/> Sure.", TOOLS) == "<stay calm/> Sure."


@pytest.mark.parametrize(
    "text", ["<speak>stay_quiet</speak>", "<speak>Stay quiet.</speak>"]
)
def test_a_tool_name_inside_ssml_is_cut(text: str) -> None:
    assert gate.judge(text, TOOLS, final=True).action == gate.CUT_AND_EXECUTE


# 11-08 08:37:56: a follow-up after express wrote its reasoning as plain text and all of it
# was spoken. Invented text in the same shape.
REASONING_1108 = (
    "<thought\n"
    "Thinking Process:\n"
    "1. Current state: The person is in the kitchen, asking about soup. "
    "2. Mood: Cheerful, a little sleepy. "
    "3. Goal: Answer briefly and warmly.\n"
    "4. Draft: Say something about soup."
)


@pytest.mark.parametrize(
    "text",
    [
        REASONING_1108,
        REASONING_1108.removeprefix("<thought\n"),
        "<thoughtThinking Process: they asked about soup.",
        "<|channel>thought\nThey asked about soup.",
        "  <THOUGHT>They asked about soup.",
        "**Thinking Process:**\n1. Current state: someone asked about soup.",
        "**Thought Process**\nThey asked about soup.",
        "## Thinking process\nThey asked about soup.",
        "thought process: they asked about soup.",
    ],
)
def test_a_reply_that_opens_as_reasoning_is_thinking(text: str) -> None:
    for final in (False, True):
        assert gate.judge(text, TOOLS, final=final) == gate.Verdict(
            gate.CUT, kind="thinking"
        )


@pytest.mark.parametrize(
    "text",
    [
        "Plan: let's go",
        "Analysis: it's rain",
        "Reasoning: it's late.",
        "Thinking:\nmaybe soup.",
        "1. Leave at 7:30",
        "1. Make coffee",
        "1. **Current state**: someone asked about soup.",
        "Thinking of pancakes",
        "I'm thinking of pancakes.",
        "In the morning",
        "An apple",
        "Thoughtful of you.",
        "Here it is: Thinking Process: soup.",
    ],
)
def test_a_reply_that_is_not_one_of_the_two_shapes_plays(text: str) -> None:
    assert gate.judge(text, TOOLS) == gate.Verdict(gate.PLAY)
    assert gate.judge(text, TOOLS, final=True) == gate.Verdict(gate.PLAY)


@pytest.mark.parametrize(
    "text",
    ["Thin", "Thou", "**Th", "# Th", "<tho", "<|ch", "Thinking", "Thinking ",
     "Thinking Pro", "Thought Process", "**Thinking Process**"],
)  # fmt: skip
def test_the_start_of_a_reasoning_opening_waits(text: str) -> None:
    assert gate.judge(text, TOOLS).action == gate.UNDECIDED
    assert gate.judge(text, TOOLS, final=True).action == gate.PLAY


@pytest.mark.parametrize(
    "text",
    ["In", "An", "Th", "1.", "#", "Thinking o", "Thinking about", "1. Make coffee"],
)
def test_a_short_or_diverging_opening_plays_at_once(text: str) -> None:
    assert gate.judge(text, TOOLS) == gate.Verdict(gate.PLAY)


def test_a_tool_code_tag_before_words_is_taken_out_of_the_kept_text() -> None:
    """11-11, 08:27 to 08:56: '<tool_code> express(...) </tool_code>' led 22 replies, none
    caught, so each was kept whole and copied by the next."""
    said = '<tool_code> express(feeling="happy", intensity=0.6) </tool_code> That was fun!'
    assert gate.without_calls(said, TOOLS).strip() == "That was fun!"


@pytest.mark.parametrize(
    ("text", "args"),
    [
        ('<tool_code> express(feeling="happy", intensity=0.6) </tool_code> Fun!',
         {"feeling": "happy", "intensity": 0.6}),
        ('<call:express{feeling:<|"|>shy<|"|>,intensity:0.4}/>Sorry.',
         {"feeling": "shy", "intensity": 0.4}),
        ("express(feeling=os.system('x'))", None),  # literals only
        ("I express myself (sometimes).", None),
        ("Hello there.", None),
    ],
)  # fmt: skip
def test_a_leaked_express_call_gives_back_its_arguments(text: str, args: Any) -> None:
    """11-11: 18 of a morning's 22 leaks were express, and none ran."""
    assert gate.leaked_express(text) == args


@pytest.mark.parametrize("said", ["I love music (jazz most).", "Dance(s) are fun.", "Look (over there)."])
def test_a_tool_name_before_brackets_in_words_plays(said: str) -> None:
    """11-11: a Python call needs no space, and empty, keyword or quoted arguments."""
    assert gate.judge(said, TOOLS, final=True).action == gate.PLAY


@pytest.mark.parametrize(
    ("text", "open_call"),
    [
        ("Sure. <tool_code> express(", True),
        ("Sure. <tool_code>", True),
        ('Sure. express{"feeling":', True),
        ("Sure. <call:express{feeling:", True),
        ('Sure. express(feeling="jo', True),
        ("Sure, sounds good.", False),
        ("I like music (jazz", False),
        ("I like to dance(s", False),
        ("I want to cal", False),
        ("Look at this `", False),
        ('Done. <tool_code> express(feeling="x") </tool_code> and more', False),
    ],
)
def test_a_reply_cut_off_inside_a_call_is_found(text: str, open_call: bool) -> None:
    """The 11-11 review, F3: a barge-in mid-tag; words cut short stay words."""
    assert gate.ends_in_open_call(text, TOOLS) is open_call
