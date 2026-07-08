# Specification: Gold Standard HTML-to-Markdown Extraction Pipeline

## 1. System Goal
To engineer a highly robust, multi-agent LLM pipeline that converts raw news blog HTML into a pristine, "gold standard" Markdown story. The system must autonomously filter out high-entropy web noise (advertisements, UI panels, navigation links) and resolve generative hallucinations by leveraging statistical consensus and semantic grounding across parallel extractions.

---

## 2. Core Algorithm & Workflow
The pipeline utilizes a **Global Arc + Constrained Tournament** architecture, heavily relying on local spatial voting to resolve discrepancies. 

### Phase 1: Parallel Extraction
* **Execution:** Run 16 independent LLM extraction calls directly on the raw HTML. 
* **Output:** 16 distinct Markdown drafts. (At this stage, drafts will contain localized hallucinations and varying degrees of captured web noise).

### Phase 2: Vector Embedding & Similarity Matrix
* **Execution:** Encode all 16 Markdown drafts into L2-normalized vector embeddings using a lightweight, locally hosted model (e.g., `all-MiniLM-L6-v2`).
* **Output:** A distance matrix calculating the cosine dissimilarity between all extractions.

### Phase 3: Mini-Max Quartet Formation
* **Execution:** Partition the 16 drafts into 4 groups of 4 (quartets) using a mini-max strategy to maximize context difference:
    1. Identify the 4 pairs of drafts with the absolute maximum mathematical distance (maximizing structural/noise variance).
    2. Assign the remaining 8 drafts to these 4 core pairs by matching each to the closest core-pair centroid.
* **Objective:** This guarantees that underlying noise and prompt-induced hallucinations are uncorrelated within the group, while providing enough local density (4 items) to surface the true signal.

### Phase 4: The 4-Way Consensus Merge Tournament
* **Execution:** Pass each quartet to an LLM Judge node. The model evaluates the 4 texts simultaneously using strict heuristic rules:
    * **Majority Rule (Statistical Weight):** Facts or structures present in 3 or 4 drafts are retained. Outliers present in only 1 draft are instantly destroyed as hallucinations.
    * **Semantic Rule:** Elements must align with the core narrative; tangent links or promotional breaks are discarded.
* **Output:** The 16 initial drafts are merged down to 4 refined drafts, which are then merged once more into the final Gold Standard text.

---

## 3. Vulnerabilities & Planned Improvements
To ensure complete mathematical and statistical independence, the prototype must be hardened against standard LLM behavioral flaws:

* **Mitigating "Lost in the Middle" (Context Permutation):**
    * *Flaw:* LLMs ignore the center of massive 40K token prompts. 
    * *Fix:* Structurally rotate or chunk the raw HTML across the 16 initial extractors so every DOM element occupies the highly-attended beginning or end of the prompt window at least once.
* **Mitigating Consensus Hallucinations (Heterogeneous Ensembling):**
    * *Flaw:* 16 identical models with identical prompts will generate correlated, identical hallucinations.
    * *Fix:* Introduce high variance in Phase 1 by using different foundation models or drastically different system prompts (e.g., Structuralist, Summarizer, Reverse-Chronological) to ensure errors remain random.
* **Mitigating Coarse Vectorization (Block-Level Alignment):**
    * *Flaw:* Dense embeddings average out the meaning of large documents, masking critical, paragraph-level discrepancies.
    * *Fix:* Chunk the 16 drafts by header or paragraph first, then run the Mini-Max pairing algorithm on specific text blocks rather than the whole document.
* **Mitigating Final-Mile Bleed-Through (NLI Entailment):**
    * *Flaw:* The tournament is a closed LLM loop; an error surviving Round 1 could theoretically make it to the end.
    * *Fix:* Add a final, deterministic Natural Language Inference (NLI) check. Compare the final Markdown sentence-by-sentence against a tag-stripped, raw text dump of the HTML. Any sentence not strictly entailed by the source text is deleted.
  