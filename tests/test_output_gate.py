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


@pytest.mark.parametrize("text", ["s", "St", "stay", "stay ", "stay_", "stay qu", "look at", "web"])
def test_the_start_of_a_multi_word_name_waits(text: str) -> None:
    assert gate.judge(text, TOOLS).action == gate.UNDECIDED
    assert gate.judge(text, TOOLS, final=True).action in (gate.PLAY, gate.PLAY_AND_LOG)


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
