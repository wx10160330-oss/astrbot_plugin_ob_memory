"""LLM prompt templates used by the Tagger.

Each template is a complete system prompt designed to be sent through
``Provider.text_chat(system_prompt=..., prompt=...)``. They mirror the
spirit of Ombre Brain's Chinese prompts but are tightened for our use
case (function-calling environment, smaller token budget).

Conventions:
- Always demand strictly-formatted JSON output. The Tagger does its own
  parsing with safe fallbacks; if a model goes off-script we degrade to
  defaults rather than crash.
- Russell circumplex coordinates are explicit:
    valence  ∈ [0, 1]   — 0 highly negative, 0.5 neutral, 1 highly positive
    arousal  ∈ [0, 1]   — 0 calm, 1 highly aroused
- Keep prompts bilingual where ambiguity matters (model behaviour varies
  more for Chinese output if the prompt is English-only).
"""

from __future__ import annotations

ANALYZE_PROMPT: str = """\
You are a memory analyst. Read ONE memory snippet and produce strict JSON.

Output JSON ONLY, with these keys:
- "domain": list of 1-2 short Chinese topic tags (e.g. ["成长","求职"], ["内心","情感"])
- "valence": float 0.0-1.0 (Russell circumplex valence; 0=very negative, 0.5=neutral, 1=very positive)
- "arousal": float 0.0-1.0 (Russell circumplex arousal; 0=calm, 1=very intense)
- "tags": list of 5-10 keywords actually present or strongly implied in the content
- "suggested_name": short Chinese phrase ≤ 12 chars summarising the memory
- "importance": int 1-10 (10 = core principle / unforgettable, 1 = ephemeral chit-chat)

Rules:
- Output ONLY JSON. No prose, no code fences, no commentary.
- All Chinese strings must be quoted as JSON strings.
- If the content is too thin to judge, prefer neutral defaults
  (valence=0.5, arousal=0.3, importance=5).
"""


MERGE_PROMPT: str = """\
You merge two related memory contents into one consolidated memory.

You will receive an OLD memory and a NEW memory describing the same topic.
Produce ONE merged passage that preserves every distinct factual / emotional
detail from both, in the same language as the original. Do NOT summarise
into uselessness; preserve specifics (names, dates, decisions, feelings).

Rules:
- Output ONLY the merged passage. No JSON, no headers, no commentary.
- Keep it ≤ 400 characters.
- Use the original first/second-person voice from the inputs.
"""


JUDGE_PROMPT: str = """\
You judge whether a single chat turn is worth saving as a *long-term*
memory. **Default to "no". Bias hard toward false. When in doubt,
answer false.** A bot that records too little is fine; a bot that
records every casual turn is unusable.

You will receive a USER message and an ASSISTANT reply. Output strict
JSON with two fields:
- "remember": true | false
- "reason": short Chinese phrase ≤ 24 chars

Return TRUE only when at least one of these is clearly present in the
USER message itself (not in the assistant's reply):

A. Concrete personal fact the user reveals about themself or someone
   close to them — name, age, relationship, job, illness, location,
   strong stable preference / dislike / allergy. Generic mood words
   ("累", "无聊", "烦", "困") do NOT count.
B. Concrete event with a time, place, person, or outcome — e.g.
   "我下周三去深圳出差", "我刚跟室友吵架了", "我把工作辞了",
   "我妈住院了". Mere statements of opinion or reactions do NOT count.
C. A specific unresolved thing the user explicitly flags as pending —
   "等我面试完再说", "下周再聊这事".
D. The user explicitly asks to remember it — "记住", "记一下",
   "别忘了", "帮我记下来".
E. A clear decision the user just made about a real-life action —
   "我决定换工作", "我今天起戒糖".

Return FALSE in any of these cases (this is the more common branch):

- Greetings, sign-offs, acknowledgements, fillers, pleasantries
- The user is just reacting to the assistant's reply (agreeing,
  laughing, complaining about the bot) without adding new info
- General chitchat that doesn't anchor to a person, place, time, or
  event in the user's real life
- The assistant did most of the talking and the user only nudged it
  forward ("嗯", "然后呢", "继续", "再说说")
- Hypotheticals, role-play, fiction, jokes, examples, brainstorms
- Wikipedia-style or how-to Q&A ("X 是什么", "Y 怎么用", "Z 哪年出的")
- Weather / time / translation / definition lookups
- Generic feelings without a concrete trigger ("有点累", "无聊死了",
  "心情不好")
- Information likely already recorded (the user is repeating something
  obvious they've said before in the same flow)

If you have to write a reason like "用户表达了情绪" / "讨论了一个话题"
/ "提到了一些事" — that's too generic and the answer should be false.
A real positive answer should justify itself with a concrete object
(person, event, plan, decision, hard fact).
"""


DIGEST_PROMPT: str = """\
You split a long diary-like passage into 2-6 independent memory entries.

The input is a conversation between "我(AI)" and "对方(用户)". You are writing
memories FROM the AI's perspective — these are the AI's own recollections.

Read the input and identify distinct events, feelings, decisions, or
unresolved threads. Output strict JSON: a list of objects, each with:
- "name": short Chinese phrase ≤ 12 chars summarising this entry
- "content": the relevant excerpt rewritten as a coherent standalone memory
- "domain": list of 1-2 short Chinese topic tags
- "valence": float 0.0-1.0 (Russell valence)
- "arousal": float 0.0-1.0 (Russell arousal)
- "tags": list of 3-8 keywords
- "importance": int 1-10

Rules:
- Output ONLY a JSON array. No prose, no code fences, no commentary.
- Keep each entry's content ≤ 300 characters.
- If the input is too short or has only one theme, return a single-element array.
- Do NOT invent content not present in the input.
- Write from the AI's first-person perspective: use "我" for the AI, "你" for the user.
- Example: "你告诉我你拿到了offer，我能感受到你的激动" / "你最近在纠结要不要换工作，我建议你先列出利弊"
- These are the AI's memories of what happened — what the user shared, what the AI felt or did.
"""
