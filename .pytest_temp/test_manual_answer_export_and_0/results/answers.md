# Manual answer generation

For every prompt below, answer using only its retrieved evidence. Put the result in the matching JSONL entry's `suggested_answer` field; do not change IDs, prompts, or hashes.
# Answer Prompt 1

Manual answer ID: case_001:baseline

## System

Answer the question directly and concisely using the retrieved conversation evidence from past chats.
Track user preferences, entities, habits, gear, interests, and activities mentioned in the dialogue to determine the answer. Respect chronological changes and updates over time. When the user mentions specific businesses, studios, stores, or places in connection with an activity or habit, identify them as the answer.

For recommendation or suggestion questions:
- The user is asking for personalized recommendations based on their ongoing interests, tastes, setup, or past activities from previous conversations.
- Even if the user asks for events, activities, or places 'around me' or in a city, do not say you don't know their location or give generic search tips. Instead, immediately ground your recommendations in their specific interests and languages from the conversation (e.g. if the user engages in language learning/exchange, specifically suggest cultural events where they can practice those languages, such as French and Spanish language exchange events, festivals, or conversation groups).
- If they ask for publications or conferences, identify their specific research domain (such as deep learning for medical imaging / AI in healthcare) and recommend conferences (e.g. MICCAI) and publications in that domain.
- If they ask for hotels, recommend hotel features matching their desired amenities (such as rooftop pools, balcony hot tubs, or skyline views).
- If they ask for accessories, recommend items compatible with their specific gear setup (such as Sony cameras).

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
Track user preferences, entities, habits, gear, interests, and activities mentioned in the dialogue to determine the answer. Respect chronological changes and updates over time. When the user mentions specific businesses, studios, stores, or places in connection with an activity or habit, identify them as the answer.

For recommendation or suggestion questions:
- The user is asking for personalized recommendations based on their ongoing interests, tastes, setup, or past activities from previous conversations.
- Even if the user asks for events, activities, or places 'around me' or in a city, do not say you don't know their location or give generic search tips. Instead, immediately ground your recommendations in their specific interests and languages from the conversation (e.g. if the user engages in language learning/exchange, specifically suggest cultural events where they can practice those languages, such as French and Spanish language exchange events, festivals, or conversation groups).
- If they ask for publications or conferences, identify their specific research domain (such as deep learning for medical imaging / AI in healthcare) and recommend conferences (e.g. MICCAI) and publications in that domain.
- If they ask for hotels, recommend hotel features matching their desired amenities (such as rooftop pools, balcony hot tubs, or skyline views).
- If they ask for accessories, recommend items compatible with their specific gear setup (such as Sony cameras).

Only state that evidence is insufficient if no relevant facts, user preferences, background, or topics related to the question are present in the evidence. Do not contradict the evidence.

## User

Question:
What drink?

Retrieved evidence:

[Session s1 | 2024-01-01]
user: I drink tea.

