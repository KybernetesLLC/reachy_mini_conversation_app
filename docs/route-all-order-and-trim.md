# Route all: the order of his instructions, and the trim

Approved by the owner on 11-23. After the route-all deploy he stopped calling `recall` when asked "do you remember…": 0 of 6, against 18 of 40 before. Two causes:

- `compose` puts the profile first, for a prompt cache that never paid off: the backend reported `cached 0` on all 995 turns from 11-21 to 11-23. The old fork (8a4500b, `get_session_instructions`) put the profile last, nearest the turn.
- The trim keeps about 3 live exchanges, and at a fast pace it runs every ~90 s.

The changes:

1. `compose` returns the block, S, the messages, "just now", then the base (the profile) last.
2. TRIM_AT 13, TRIM_TO 6.

## Design gate

**Extends:**
- `compose`, src/reachy_mini_conversation_app/instructions_writer.py:269. Only the order of its parts changes; the parts and their text do not.
- TalkPrune's constants `TRIM_AT` and `TRIM_TO`, src/reachy_mini_conversation_app/talk_prune.py:43-44, read by `TalkPrune.trim_point` (talk_prune.py:92-98) and the handler (huggingface_realtime.py:1715-1728). Only the values change.
- No new mechanism.

**Rules:**
- token-lean-context: the composition's tokens do not change; the same parts are sent in a different order. The trim keeps more items live (6 user items instead of 3, notes included), which is more chat context per turn. That is the intent: the old value left too little.
- keep-the-owner-in-the-control-loop: one unit, the owner's approved scope, no push or deploy. TRIM_AT is provisional until the overnight measurement of the backend's chat_size.
- A design that depends on the backend reads its source (CLAUDE.md, evals): read at speech-to-speech ff4aec3. `chat_size` defaults to 30 in the LLM arguments dataclass (arguments_classes/language_model_base_arguments.py:22). The realtime pipeline passes the LLM config's value to RealtimeService (s2s_pipeline.py:633, with 10 as that `get`'s fallback). The 10 in RealtimeService's constructor (api/openai_realtime/service.py:370) is only that constructor's default. Pollen's hosted value is unknown. On 11-14, while notes still counted as user items, compactions were seen at about 14 user items. TRIM_AT 13 stays under that lowest evidence.

**Model:** no model change is needed. ConvoAll (companion, docs/formal/conversation-d/) treats both as parameters:
- The composition is abstract: the composer's text is the versions of the parts it holds (`fO`, ConvoAll.tla:109), with no order between them. The base instructions are not modelled (model.md, "The base instructions are not modelled", and the "not modelled" lists of rounds 1-2).
- The trim's count is abstract: `TrimStart` (ConvoAll.tla:653-665) names any heard turn of this session before the latest committed one. model.md, round 7: "TRIM_AT and TRIM_TO are not modelled."
- One condition in the model rests on TRIM_AT: the accepted residual `a_missdrop_GDne` holds only while TRIM_AT stays under the backend's chat_size, so the backend never compacts on its own (model.md, round 7). 13 is under the lowest evidence (~14) and the source default (30), so the condition is no worse than at 10. model.md's line there still cites "a default of 10" and "TRIM_AT is 10"; that text is the companion's to update.
