# Python AI Learning Projects

this repo is just me messing around with embeddings trying to figure out how ai.
tl;dr its very fun.

## Semantic & BM25 Search Demo

- First 2 projects were just to get the grasp of embeddings and vector dbs.
  - in the first project i just got an article dataset from hugging face and made a small semantic search on it, where i just embedded the texts and querys and found closest matches with cosine similarity.
  - in the second project i took the existing code and i added bm25 ranking algorithm with fuzzy matching for typo tolerance and better results.

---

# RAG on PDF (C# in a Nutshell)

_yea i couldn't find anything else document i had on my laptop_

- here i took what i learned from the previous two experiments and i combined it with an llm to actually get a more tangable result.
- i just took pdf read it and then chunked by sentences with 2 sentence overlap. afterwards i just run semantic search on it and it comes back with 3 closest matches which then it gives to an llm as context with a prompt that you gave. thus llm gives you a correct answer.

---

## Learning Outcomes

- Understand and implement embeddings, vector DBs
- Use embeddings for retrieval tasks
- Integrate with vector databases and LLM APIs

---

_i have used hugging face for datasets and models_
