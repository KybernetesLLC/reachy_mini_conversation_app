"""The output gate's tool-name rule (the planning session, 2026-10-17).

The cases are generated from tool registries: every tool times every shape it could
leak in. The artifact rule's own table is SPOKEN_ARTIFACT_CASES in
test_huggingface_realtime.py.
"""

from __future__ import annotations

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
    '{{"name": "{n}", "arguments": {{}}}}',
    "[{n}()]",
    "```tool_code\n{n}()\n```",
    "{n}{{}}",  # bare, no "call:" (2026-10-17, 2026-10-18)
    '{n}{{reason:<|"|>Conversational acknowledgement.<|"|>}}',
)


def _expected_cut(name: str) -> str:
    return gate.CUT_AND_EXECUTE if name in LEAK_SAFE else gate.CUT


@pytest.mark.parametrize("name", REGISTRY)
@pytest.mark.parametrize("shape", MARKUP_SHAPES)
def test_every_tool_in_every_markup_shape_is_cut(name: str, shape: str) -> None:
    verdict = gate.judge(shape.format(n=name), TOOLS, final=True)
    assert (verdict.action, verdict.tool, verdict.shape) == (_expected_cut(name), name, "markup")
    assert verdict.kind == "leaked tool call"


@pytest.mark.parametrize("name", REGISTRY)
@pytest.mark.parametrize("shape", ("{n}", "{n}.", " {n}! ", "`{n}`", '"{n}"'))
def test_every_tool_said_bare_is_judged_on_the_whole_reply(name: str, shape: str) -> None:
    """An underscore name is cut; a one-word name plays, logged; mid-reply, either waits
    or plays, never cuts.
    """
    text = shape.format(n=name)
    verdict = gate.judge(text, TOOLS, final=True)
    if "_" in name:
        assert (verdict.action, verdict.tool, verdict.shape) == (_expected_cut(name), name, "underscore name")
        assert gate.judge(text, TOOLS).action == gate.UNDECIDED
    else:
        assert (verdict.action, verdict.tool, verdict.shape) == (gate.PLAY_AND_LOG, name, "plain word")
        assert gate.judge(text, TOOLS).action in (gate.PLAY, gate.UNDECIDED)


@pytest.mark.parametrize("name", [n for n in REGISTRY if "_" in _spoken(n).strip() or " " in _spoken(n)])
@pytest.mark.parametrize("shape", ("{w}", "{W}.", "{w}!", "{h}"))
def test_every_multi_word_tool_said_as_words_is_cut(name: str, shape: str) -> None:
    words = _spoken(name)
    text = shape.format(w=words, W=words.capitalize(), h=words.replace(" ", "-"))
    verdict = gate.judge(text, TOOLS, final=True)
    assert (verdict.action, verdict.tool, verdict.shape) == (_expected_cut(name), name, "spoken name")
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


@pytest.mark.parametrize("text", ["Sta", "stay", "stay ", "stay_", "stay qu", "look at", "web", "Why"])
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
    assert gate.RULES == (gate.leaked_tool_name, gate.artifact_shape)


def test_with_no_registry_the_artifact_rule_alone_decides() -> None:
    empty = gate.ToolNames()
    assert gate.judge("call:stay_quiet{}", empty, final=True) == gate.Verdict(gate.CUT, kind="leaked tool call")
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
LEADING = f"<call:express{{feeling:{Q}shy{Q},intensity:0.4}}/>I'm sorry I'm being glitchy."


def test_a_call_left_open_at_the_end_of_the_reply_is_cut() -> None:
    assert gate.judge(OPEN_IDLE, TOOLS, final=False).action == gate.UNDECIDED
    verdict = gate.judge(OPEN_IDLE, TOOLS, final=True)
    assert (verdict.action, verdict.tool, verdict.shape) == (gate.CUT_AND_EXECUTE, "idle_do_nothing", "markup")


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
        ("Sure! <call:express{emotion:happy} That's great, 5 > 3 right?", "Sure! That's great, 5 > 3 right?"),
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


def test_an_open_tag_call_waits_while_streaming_and_ends_at_its_arguments_once_words_follow() -> None:
    head = f"<call:express{{feeling:{Q}shy{Q}}}"
    assert gate.spoken_artifact(head) == gate.UNDECIDED
    assert gate.spoken_artifact(head + " /") == gate.UNDECIDED
    assert gate.spoken_artifact(head + " />") == "leaked tool call"
    assert gate.spoken_artifact(head + " Hello") is None  # words: a reply


# 2026-10-20 22:59: "express{feeling:<|"|>delight<|"|>}" was heard. A one-word tool name was
# released before its "{" arrived; now the name, and the start of it, waits (the owner).
BARE_EXPRESS = f"express{{feeling:{Q}delight{Q}}}"


@pytest.mark.parametrize("cut_at", [3, 5, 7, 8, 12, len(BARE_EXPRESS) - 1])
def test_a_one_word_tool_name_waits_for_what_follows(cut_at: int) -> None:
    assert gate.judge(BARE_EXPRESS[:cut_at], TOOLS).action == gate.UNDECIDED


def test_the_2259_bare_express_is_cut_once_its_arguments_close() -> None:
    verdict = gate.judge(BARE_EXPRESS, TOOLS)
    assert (verdict.action, verdict.tool, verdict.shape) == (gate.CUT, "express", "markup")
    assert gate.judge(BARE_EXPRESS, TOOLS, final=True).action == gate.CUT


@pytest.mark.parametrize("text", ["Express yourself", "Expressive", "Music is nice", "Looking good"])
def test_a_sentence_that_starts_like_a_tool_name_plays_at_the_next_word(text: str) -> None:
    assert gate.judge(text, TOOLS).action == gate.PLAY
