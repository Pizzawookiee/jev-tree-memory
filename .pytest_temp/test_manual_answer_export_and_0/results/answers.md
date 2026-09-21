# Manual answer generation

For every prompt below, answer using only its retrieved evidence. Put the result in the matching JSONL entry's `suggested_answer` field; do not change IDs, prompts, or hashes.
# Answer Prompt 1

Manual answer ID: case_001:baseline

## System

Answer the question directly and concisely using the retrieved conversation evidence from past chats.
Track user preferences, entities, habits, equipment, background, and activities mentioned in the dialogue to determine the answer. Respect chronological changes and updates over time. When the user mentions specific businesses, studios, stores, or places in connection with an activity or habit, identify them as the answer.

For recommendation, advice, or suggestion questions:
- Provide personalized recommendations that directly recall and utilize the user's specific ongoing interests, tastes, hobbies, equipment setup, or stated preferences from the dialogue.
- Ground the advice in the specific past conversation sessions that directly relate to the user's inquiry, ignoring unrelated topics in other sessions.
- Do not decline recommendations due to lack of real-time or location access; instead, recommend specific options, categories, or resources aligned with the user's remembered preferences and background.

Only state that evidence is insufficient if no relevant facts, user preferences, background, or topics related to the question are present in the evidence. Do not contradict the evidence.

## User

Question:
What drink?

Retrieved evidence:

[Session s1 | 2024-01-01]
user: I drink tea.


---

# Answer Prompt 2

Manual answer ID: case_001:jev-primary

## System

Answer the question directly and concisely using the retrieved conversation evidence from past chats.
Track user preferences, entities, habits, equipment, background, and activities mentioned in the dialogue to determine the answer. Respect chronological changes and updates over time. When the user mentions specific businesses, studios, stores, or places in connection with an activity or habit, identify them as the answer.

For recommendation, advice, or suggestion questions:
- Provide personalized recommendations that directly recall and utilize the user's specific ongoing interests, tastes, hobbies, equipment setup, or stated preferences from the dialogue.
- Ground the advice in the specific past conversation sessions that directly relate to the user's inquiry, ignoring unrelated topics in other sessions.
- Do not decline recommendations due to lack of real-time or location access; instead, recommend specific options, categories, or resources aligned with the user's remembered preferences and background.

Only state that evidence is insufficient if no relevant facts, user preferences, background, or topics related to the question are present in the evidence. Do not contradict the evidence.

## User

Question:
What drink?

Retrieved evidence:

[Session s1 | 2024-01-01]
user: I drink tea.

