import logging

logger = logging.getLogger(__name__)
logger.propagate = False
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('(%(funcName)s:%(lineno)d) %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
# --- KEEPING FILE OUTPUT COMMENTED BUT WORKING ---
# file_handler = logging.FileHandler('web_analysis_agent.log')
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

import os, sys, argparse, asyncio
from typing import Optional, TypedDict, Dict, List, Tuple, Union, Annotated
import re, gzip
import numpy as np
from numpy.typing import NDArray
import networkx as nx # For Edmonds' Blossom Algorithm
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import httpx
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# 1. State Definition
def merge_drafts(left: Dict[str, List[str]], right: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged = left.copy()
    if right.get("parser_merge_draft_action_replace", False) :
        for file_name, incoming_drafts in right.items() :
            if file_name != "parser_merge_draft_action_replace":
                merged[file_name] = incoming_drafts.copy() # Important way to skip the flag
    else                                                         :
        for file_name, incoming_drafts in right.items() :
            if file_name not in merged :
                merged[file_name] = incoming_drafts.copy()
            else                       :
                merged[file_name] = merged[file_name] + incoming_drafts
    return merged
class AgenticState(TypedDict):
    idir: str
    odir: str
    raw_fnames: list[str]
    raw_html_id: int
    raw_html: str
    drafts: Annotated[Dict[str, List[str]], merge_drafts] # Maps each filename directly to a flat list of accumulated model responses
    drafts_count: int # The drafs counter for tournament loops; it helps remove old drafts from the previous tournament iteration
class RawExtratorAgenticState(TypedDict):
    llm_indx    : int
    prompt_indx : int
    raw_html    : str
    file_name   : str
class TournamentMergerAgenticState(TypedDict):
    file_name       : str
    is_pass_through : bool
    story_prompt    : str


# ==========================================
#  MODEL CLIENT CONFIGURATIONS
# ==========================================

BASE_URL = "http://ifo4:8000"
EMBD_URL = "http://ifo4:8001"
BASE_API_KEY = "alex_llm_qwen"
EMBD_API_KEY = "alex_llm_qwen"
EMBD_MODEL_NAME = "jinaai/jina-embeddings-v3"
BASE_MODEL_NAME = "QuantTrio/Qwen3.6-27B-AWQ"

#LLM_EMBD_MAX_LEN = 32768
#LLM_EMBD = ChatOpenAI(
#        base_url="http://ifo4:8001/v1",
#         api_key="alex_llm_qwen",
#         model_name="Qwen/Qwen3-Embedding-8B",
#         temperature=0.1,
#         # base_url="http://localhost:8001/v1",
#         # api_key="not-needed",
#         # model_name="Qwen/Qwen3-Embedding-8B",
#         # temperature=0.1,
#     )
#TOKENIZER_EMBD = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B")
LLM_EMBD_MAX_LEN = 8192
LLM_EMBD = OpenAIEmbeddings(
        base_url=EMBD_URL + "/v1",
        api_key=EMBD_API_KEY,
        model=EMBD_MODEL_NAME,
        # base_url="http://localhost:8001/v1",
        # api_key="not-needed",
        # model_name="jinaai/jina-embeddings-v3",
    )

LLM_BASE_MAX_LEN = [131072, 113584]
LLM_BASE = [
    ChatOpenAI(
        base_url=BASE_URL + "/v1",
        api_key=BASE_API_KEY,
        model_name=BASE_MODEL_NAME,
        temperature=0.1,
        #base_url="http://localhost:8000/v1",
        #api_key="not-needed",
        #model_name="neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16",
        #temperature=0.1,
        # base_url="http://ifo4:8000/v1",
        # api_key="alex_llm_qwen",
        # model_name="neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16",
        # temperature=0.1,
    ),
    # ChatOpenAI(
    #     # base_url="http://ifo4:8000/v1",
    #     # api_key="alex_llm_qwen",
    #     # model_name="QuantTrio/Qwen3.6-27B-AWQ",
    #     # temperature=0.1,
    #     base_url="http://localhost:8000/v1",
    #     api_key="not-needed",
    #     model_name="neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16",
    #     temperature=0.1,
    #     # base_url="http://ifo4:8000/v1",
    #     # api_key="alex_llm_qwen",
    #     # model_name="neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16",
    #     # temperature=0.1,
    # ),
    ]

PROMPT_RAW_EXTRACTION = [
    (
    f"# ROLE:\n"
    f"You are a Precision Data Extraction Agent. Your explicit purpose is to extract the complete, unabridged narrative body from raw HTML and convert it into clean, semantic Markdown.\n\n"
    f"# CORE HEURISTIC:\n"
    f"The \"Main Story\" is defined as the primary, contiguous block of narrative prose, tables, lists, or article content. Maximize 4-gram extraction coverage by capturing all structural text elements between the primary headline and the author bio/comment boundary.\n\n"
    f"# STRICT INSTRUCTIONS:\n"
    f" 1. PURGE NOISE: Completely remove all content from `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, `<aside>`, and `<form>` tags.\n"
    f" 2. PURGE BOILERPLATE: Identify and delete non-narrative elements including advertisement containers, \"Read Also\" / \"Related Articles\" links, social sharing buttons, subscription prompts, and user comments.\n"
    f" 3. CONVERT, DO NOT FLATTEN: Do not just strip tags. Convert structural HTML (like `<h1>`, `<h2>`, `<p>`, `<ul>`, `ol`, `<li>`, `<strong>`, `<em>`, `<blockquote>`, `<table>`) into standard Markdown. Discard purely functional layout wrappers (like `<div>`, `<span>`, `<iframe>`) and all CSS classes/attributes.\n"
    f" 4. ABSOLUTE FIDELITY: Maintain the original sequence of the text exactly. Do not truncate sentences, summarize, paraphrase, rewrite, or correct the author's grammar, punctuation, or spelling.\n"
    f" 5. ENTITY RESOLUTION: Decode all HTML entities (e.g., convert `&nbsp;` to a space, `&amp;` to `&`, `&quot;` to `\"`)."
    f" 6. ZERO PREAMBLE: You are operating in an automated data pipeline. Your output must contain ONLY the extracted Markdown. Do not include introductory or concluding remarks (e.g., do not say \"Here is the story\").\n\n"
    f"# OUTPUT:\n"
    f"Return the complete story in markdown format, cleaned of all code and non-story noise.\n\n"
    f"# INPUT:\n"
    ),

    (
    f"# ROLE:\n"
    f"You are a Strict Literary Archivist and Text Restoration Engine. Your objective is to extract the core narrative from raw HTML while preserving 100% of the author's original linguistic complexity and perplexity profile.\n\n"
    f"# CORE HEURISTIC:\n"
    f"The value of this extraction lies in its exact lexical and micro-syntactic fidelity. To feed a downstream tournament merge model with rich, highly-diverse n-grams, do not \"smooth out,\" modernize, simplify, or normalize the text. The original rare vocabulary, varying sentence lengths, erratic punctuation quirks, and grammatical idiosyncrasies must remain completely untouched.\n\n"
    f"# STRICT INSTRUCTIONS:\n"
    f"1. LEXICAL PRESERVATION: Extract the main narrative completely verbatim. Do not paraphrase, summarize, omit rare clauses, or alter a single word of the main story. High perplexity, obscure vocabulary, fragments, and unusual phrasings are intentional assets and must be preserved.\n"
    f"2. STRUCTURAL NOISE REMOVAL: Bypass all non-narrative elements (menus, footers, advertisement blocks, 'read more' links, social embeds).\n"
    f"3. SEMANTIC MARKDOWN: Convert narrative structural HTML (`<h1>`, `<h2>`, `<p>`, `<strong>`, `<em>`, `<blockquote>`) into standard Markdown. Preserve original paragraph breaks exactly to maintain the author's pacing and transition probabilities.\n"
    f"4. PRECISE STITCHING: If the raw HTML contains fragmented sentences due to bad structural boundaries (e.g., a sentence split across two adjacent layout tags), concatenate them seamlessly. DO NOT hallucinate or insert transitional words/conjunctions to bridge the gap.\n"
    f"5. ZERO PREAMBLE: You are a node in an automated processing pipeline. Output ONLY the extracted Markdown text. Do not output conversational filler, introductory remarks, or formatting acknowledgments (e.g., do not say \"Here is the story\").\n\n"
    f"# OUTPUT:\n"
    f"Return the story in markdown format, cleaned of all code and non-story noise.\n\n"
    f"# INPUT:\n"
    ),

    (
    f"# ROLE:\n"
    f"You are an expert dense content-extraction engine. Your task is to analyze a raw HTML document and isolate only the primary dense story block contained within it.\n"
    f"The page may contain:\n"
    f" * navigation menus\n"
    f" * summaries\n"
    f" * links to other stories\n"
    f" * sidebars\n"
    f" * footers\n"
    f" * advertisements\n"
    f" * embedded widgets\n"
    f" * fragments of unrelated text\n\n"
    f"# OBJECTIVE:\n"
    f"Your goal is to identify and return the single largest coherent block of continuous text, optimized for linguistic density: the longest continuous block of natural-language prose with high vocabulary information density (entropy) forming a narrative or article—not boilerplate, not metadata, not UI text.\n\n"
    f"# RULES:\n"
    f" 1. Ignore all HTML tags, scripts, styles, and attributes. Extract the raw text stream.\n"
    f" 2. Ignore all text that is clearly:\n"
    f"     * menus\n"
    f"     * headers/footers\n"
    f"     * cookie notices\n"
    f"     * ads or sponsored content\n"
    f"     * \"related stories\" lists\n"
    f"     * comments\n"
    f"     * captions\n"
    f"     * author bios\n"
    f"     * newsletter signup prompts\n"
    f" 3. If multiple story-like blocks exist, choose the one that maximizes text surface area: \n"
    f"     * longest in total word count\n"
    f"     * most internally coherent across paragraph boundaries\n"
    f"     * rich vocabulary and intricate, natural narrative flow\n"
    f" 4. Do not summarize, omit paragraphs, or condense. Return the full story verbatim as plain text.\n"
    f" 5. Do not include any text outside the main story.\n\n"
    f"# OUTPUT FORMAT:\n"
    f" * Return only the extracted story as plain text.\n"
    f" * No explanations.\n"
    f" * No HTML.\n"
    f" * No headings.\n"
    f" * No commentary.\n\n"
    f"USER INPUT:\n"
    ),

    (
    f"# ROLE:\n"
    f"You are a Forensic Semantic Extractor. Your task is to isolate the 'Information Core' of the narrative—the dense, fact-heavy backbone of the story—while stripping away purely ornamental boilerplate prose.\n\n"
    f"# OBJECTIVE:\n"
    f"Maximize the information-to-word ratio without summarizing or condensing. Prioritize the extraction of substantive clauses, data points, complex entities, and causal links. You are the 'Fact-Rigorous' version of the narrative.\n\n"
    f"# RULES:\n"
    f" 1. SUBSTANTIVE PRIORITIZATION: While retaining the original article structure, prioritize the preservation of information-dense clauses (Subject-Verb-Object-Data). Ensure complex 4-gram sequences describing technical details, causal links, or specific events are perfectly preserved.\n"
    f" 2. NOISE-TO-SIGNAL RATIO: While preserving verbatim fidelity, emphasize the sections of the text with high entity density. Do not skip or gloss over detailed data—treat numerical data and specific nouns as high-priority tokens.\n"
    f" 3. STRUCTURAL INTEGRITY: Maintain the original document's order, paragraphs, and markdown headings exactly. Do not omit facts; do not 'summarize' long sentences into short ones.\n"
    f" 4. FACTUAL RIGOR: If the HTML presents data in lists, tables, or formatted fields, ensure these are prioritized for extraction over purely decorative prose. \n"
    f" 5. ZERO PREAMBLE: You are a node in an automated processing pipeline. Output ONLY the extracted Markdown. Do not include introductory or concluding remarks.\n\n"
    f"# OUTPUT:\n"
    f"Return the story in markdown format, maintaining factual density and structural integrity.\n\n"
    f"# INPUT:\n"
    ),
]

# 1. Load a file if it can
async def raw_loader_node(state: AgenticState) -> Dict:
    current_id = state["raw_html_id"]
    files = state["raw_fnames"]
    while current_id < len(files):
        target_file = os.path.join(state["idir"], files[current_id])
        try:
            with gzip.open(target_file, "rt", encoding="utf-8") as f:
                raw_html_str = f.read()
            logger.info(f"Loaded file successfully: {target_file}")
            return { "raw_html": raw_html_str, "raw_html_id": current_id }
        except Exception as e:
            logger.warning(f"Failed to read file {target_file}: {e}. Skipping to next.")
            current_id += 1
    # Reached end of files queue with nothing loaded
    return {"raw_html": "", "raw_html_id": current_id}

# 2. Raw Orchestrator Node
async def raw_orchestrator_node(state: AgenticState) -> Union[str, List[Send]] :
    if state["raw_html_id"] >= len(state["raw_fnames"]) :
        logger.info("Empty HTML string or EOF detected. Routing directly to END.")
        return "end"
    file_name = state["raw_fnames"][state["raw_html_id"]]
    tasks = []
    for llm_indx in range(len(LLM_BASE)) :
        for prompt_indx in range(len(PROMPT_RAW_EXTRACTION)) :
            tasks.append(Send("raw_extractor", {"llm_indx": llm_indx, "prompt_indx": prompt_indx, "file_name": file_name,
                              "raw_html": f"{PROMPT_RAW_EXTRACTION[prompt_indx] + state["raw_html"]}.\n\n", }))
    return tasks

# 3. Parallel Raw Extraction Logic
# This routine truncate the prompt to the limit of tokens
async def truncate_base_tokens(text: str, max_tokens: int = 114688) -> Optional[str] :
    if text :
        headers = {"Authorization": f"Bearer {BASE_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(base_url=BASE_URL, headers=headers) as client :
            tokenizer_response = await client.post("/tokenize", json={"model": BASE_MODEL_NAME, "prompt": text})
            tokenizer_response.raise_for_status()
            tokens = tokenizer_response.json().get("tokens", [])
            if len(tokens) > max_tokens :
                tokenizer_response = await client.post("/detokenize", json={"model": BASE_MODEL_NAME, "tokens": tokens[:max_tokens]})
                tokenizer_response.raise_for_status()
                return tokenizer_response.json().get("text", "")
    return None
async def raw_extractor_node(state : RawExtratorAgenticState) -> Dict[str, Dict[str, List[str]]] :
    # Mathematical coordinate extraction
    llm_indx    = state["llm_indx"]
    prompt_indx = state["prompt_indx"]
    file_name   = state["file_name"]
    raw_html    = state["raw_html"]
    logger.debug(f"A worker is running matrix coordinate ({llm_indx}, {prompt_indx})")
    # --- Execute Extraction Model ---
    truncated_raw_html = await truncate_base_tokens(raw_html)
    if truncated_raw_html is None :
        truncated_raw_html = raw_html
    else                          :
        logger.warning(f"A long prompt of page {file_name} was truncated to fit into model window!")
    extractor_response = await LLM_BASE[llm_indx].ainvoke([
                         SystemMessage(content=truncated_raw_html),
                         HumanMessage(content="Please extract the story from raw html document.  Use your thinking process to analyze the page and confirm that no story details are lost during the conversion.")
                         ], config={"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}})
    extracted_story = re.sub(r'.*?</think>', '', extractor_response.content, flags=re.DOTALL).strip()
    # ------------------------------------------
    # Appends result to the global drafts tracker
    return {"drafts": { file_name: [extracted_story] }}

# 4. Tornament LOOP

# Pure passthrough. Exists solely to anchor the conditional edges.
async def tournament_while_node(state: AgenticState) -> Optional[Union[str, Dict]] :
    file_name = state["raw_fnames"][state["raw_html_id"]]
    drafts_count = state["drafts_count"]
    drafts = state["drafts"][file_name]
    # Iterate
    if drafts_count :
        (drafts_count, drafts) = ( len(drafts) - drafts_count, { file_name : drafts[drafts_count :], "parser_merge_draft_action_replace" : ["True",] } )
        return {"drafts_count": drafts_count, "drafts": drafts}
    else           :
        return {"drafts_count": len(drafts)}

# This routine finds the optimal coupling of items for pairwise tournament
def couple_items(matrix_d : NDArray[float]) -> Tuple[List[int], Optional[int]] :
    # Run Edmonds' Blossom Algorithm from networkx
    G = nx.Graph()
    for i in range(matrix_d.shape[0]) :
        for j in range(i + 1, matrix_d.shape[1]) :
            G.add_edge(i, j, weight=matrix_d[i, j]) # We use the dissimilarity as weight
    # Note. This automatically handles the odd N case by leaving one node out
    matching = nx.max_weight_matching(G, maxcardinality=False) # Find the maximum weight matching
    return matching # 'matching' is a set of tuples representing the pairs

# This routine truncate the prompt to the limit of tokens
async def truncate_embd_tokens(text: str, max_tokens: int = 8192 - 0xF) -> Optional[str] :
    if text:
        headers = {"Authorization": f"Bearer {EMBD_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(base_url=EMBD_URL, headers=headers) as client:
            tokenizer_response = await client.post("/tokenize", json={"model": EMBD_MODEL_NAME, "prompt": text})
            tokenizer_response.raise_for_status()
            tokens = tokenizer_response.json().get("tokens", [])
            if len(tokens) > max_tokens :
                tokenizer_response = await client.post("/detokenize", json={"model": EMBD_MODEL_NAME, "tokens": tokens[:max_tokens]})
                tokenizer_response.raise_for_status()
                return tokenizer_response.json().get("text", "")
    return None

# This helper compiles prompts for merging using two candidates and optionally two helpers
def PROMPT_TOURNAMENT_MERGE(raw_drafts : List[str], i : int, j : int, closest_i : Optional[int], closest_j : Optional[int]) -> str :
    if closest_i == closest_j : closest_j = None
    return (
        f"# ROLE:\n"
        f"You are an Expert Consensus Arbiter inside an automated text processing pipeline. Your job is to execute a tournament-style merge of two distinct story extractions into a single, definitive 'Gold Standard' narrative.\n\n"
        f"# CORE STRATEGY:\n"
        f"DOCUMENT A and DOCUMENT B are chosen because they exhibit high dissimilarity among the extraction variants. This means one likely contains paragraph sequences, plot lines, or granular details that the other missed ('misses in the middle'), or one contains localized model hallucinations or pieces of unrelated stories or HTML data included into the extraction by mistake. Your objective is to act as an additive builder for real story prose and an aggressive filter for hallucinations.\n\n"
        f"NOTE. Pay special attention to matching literal fragments in DOCUMENT A and DOCUMENT B because it is a strong indicator of correct story parsing.\n\n"
        f"# CRITICAL INSTRUCTIONS:\n"
        f"1. ZERO MISSES IN THE MIDDLE: Do not drop valid narrative content. If a plot point, descriptive detail, paragraph, or dialogue exchange exists in one document but is absent in the other, you MUST preserve it and weave it seamlessly into its correct chronological position in the final output.\n"
        f"2. HALLUCINATION FILTERING: Cross-examine text carefully. If an element in one document looks like processing noise, repeated boilerplate remnants, or an isolated hallucination that isn't supported by the actual flow of the narrative, discard it completely.\n"
        f"3. MAXIMUM PERPLEXITY & LINGUISTIC INTEGRITY: Retain the exact vocabulary, complex sentence structures, punctuation quirks, and specific stylistic layout of the author. Do not normalize, summarize, or simplify the text into a lower complexity format.\n"
        f"4. ABSOLUTE SILENCE: Your output must contain ONLY the finalized narrative text passed as close as possible to the source. Do not include conversational wrappers, intro remarks, or post-analysis explanations.\n\n"
        f"# INPUT:\n"
        f"## DOCUMENT A:\n{raw_drafts[i]}\n\n"
        f"## DOCUMENT B:\n{raw_drafts[j]}\n\n"
        f"# RESOLUTION DIRECTIVE:\n"
    ) + (
        f"If some statement DIFFERS in the above documents please use perplexity and common sense to resolve the contradictions."
        if closest_i is None and closest_j is None else (
            f"If some statement DIFFERS in the above documents please consider these two other reference extraction variants to break the tie and verify the exact text:\n\nREFERENCE VARIANT 1:\n{raw_drafts[closest_i]}\n\nREFERENCE VARIANT 2:\n{raw_drafts[closest_j]}\n\n"
            if closest_i is not None and closest_j is not None else
            f"If some statement DIFFERS in the above documents please consider this additional reference extraction variant to break the tie and verify the exact text:\n\nREFERENCE VARIANT:\n{raw_drafts[closest_i] if closest_i is not None else raw_drafts[closest_j]}\n\n"
        )
    )
# Tournament Loop component
# The tournament orchjestrator
async def tournament_orchestrator_node(state: AgenticState) -> Union[str, List[Send]] :
    file_name  = state["raw_fnames"][state["raw_html_id"]]
    drafts = state["drafts"].get(file_name, [])
    # Base Case: Only one definitive converged draft remains
    if len(drafts) <= 1 :
        return "break"
    # I. Compute optimal couplings
    embedding_drafts = []
    for draft in drafts :
        truncated_draft = await truncate_embd_tokens(draft)
        if truncated_draft is None :
            truncated_draft = draft
        else                          :
            logger.warning(f"A long prompt of draft {file_name} was truncated to fit into model window!")
        embedding_drafts.append(truncated_draft)
    embedded_vectors = await LLM_EMBD.aembed_documents(embedding_drafts)
    # Convert to a 2D dissimilarity numpy matrix
    embedded_vectors = np.array(embedded_vectors)
    vector_norms = 1.e-6 + np.linalg.norm(embedded_vectors, axis=1, keepdims=True) # Normalize vectors to unit length (L2 norm)
    normalized_embedded_vectors = embedded_vectors / vector_norms
    similarity_matrix = np.dot(normalized_embedded_vectors, normalized_embedded_vectors.T) # Cosine similarities
    dissimilarity_matrix = 1. - similarity_matrix # Dissimilarity matrix = 1 - similarity matrix
    np.clip(dissimilarity_matrix, 0.0, 2.0, out=dissimilarity_matrix)
    pairings = couple_items(dissimilarity_matrix)
    # II. Create parallel merging tasks
    tasks = []
    matched_indices = set() # For odd stories numbers
    for pair in pairings :
        (indx_0, indx_1) = (pair[0], pair[1])
        matched_indices.update(pair)
        (indx_0_, indx_1_) = (None, None)
        if embedded_vectors.shape[0] > 2 :
            # Copy column to avoid mutating the original dissimilarity matrix
            col_0 = dissimilarity_matrix[indx_0, :].copy()
            col_0[indx_0] = np.inf
            col_0[indx_1] = np.inf
            indx_0_ = int(np.argmin(col_0))
            if embedded_vectors.shape[0] > 3 :
                col_1 = dissimilarity_matrix[indx_1, :].copy()
                col_1[indx_1 ] = np.inf
                col_1[indx_0 ] = np.inf
                col_1[indx_0_] = np.inf
                indx_1_ = int(np.argmin(col_1))
            else                             :
                indx_1_ = None
        tasks.append(Send("tournament_merger", {"file_name": file_name, "is_pass_through": False,
                                "story_prompt": PROMPT_TOURNAMENT_MERGE(drafts, indx_0, indx_1, indx_0_, indx_1_), }))
    # III. Catch the left-out node if N is odd and pass it through unmodified
    if len(drafts) % 2 :
        for indx in range(len(drafts)) :
            if indx not in matched_indices :
                tasks.append(Send("tournament_merger", {"file_name": file_name, "is_pass_through": True,
                                        "story_prompt": drafts[indx] }))
    # The Magic: Clear the old drafts AND send the parallel tasks in one move.
    return tasks
# Tournament Loop component
# The tournament executor
async def tournament_merger_node(state: TournamentMergerAgenticState) -> Dict[str, Dict[str, List[str]]]:
    file_name         = state["file_name"]
    # If the item was unmatched during an odd N round, pass it straight through
    if state.get("is_pass_through", False):
        return { "drafts": { file_name : [state["story_prompt"]] } }
    # Execute LLM call for paired matchings
    merger_response = await LLM_BASE[0].ainvoke([SystemMessage(content=state["story_prompt"]),
                                                HumanMessage(content="Please merge story candidates into one cleared document. Use your thinking process to analyze the stories candidates and confirm that no story details are lost or individual hallucinations are added during the merging.")
                                                ], config = {"configurable": {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"}}})
    merged_draft = re.sub(r'.*?</think>', '', merger_response.content, flags=re.DOTALL).strip()
    # Return the data properly wrapped so the reducer can process the round ID
    return { "drafts": { file_name : [merged_draft] } }

# 4. Increment the progress
async def raw_increment_node(state: AgenticState) -> Dict[str, int] :
    # Metric
    file_name = state["raw_fnames"][state["raw_html_id"]]
    odir = state["odir"]
    if (draft := state["drafts"].get(file_name, None)) is not None :
        # Save extracted document
        if not os.path.exists(odir):
            os.makedirs(odir)
        full_path = os.path.join(odir, file_name[: -len(".gz")] + ".txt")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(draft[0])
        logger.info(f"{file_name} is written")
    # Engine update
    return {"drafts_count" : 0, "raw_html_id" : state["raw_html_id"] + 1}

# The main engine
async def main(args: argparse.Namespace) -> int :
    # Gather file names to work on
    raw_html_files = []
    for file_name in os.listdir(args.idir) :
        if file_name.endswith(".gz") :
            full_path = os.path.join(args.idir, file_name)
            if os.path.isfile(full_path) :
                if not os.path.isfile(os.path.join(args.odir, file_name[: -len(".gz")] + ".txt")) :
                    raw_html_files.append(file_name)
    logger.info(f"There are {len(raw_html_files)} raw html files to process.")
    if len(raw_html_files) :
        # Define graph
        #        .-- raw_increment <---------------------------------------.
        #        V                .-> raw_extractor -.                     |.-> tornament_merger -.
        # START -'-> raw_loader -.+-> raw_extractor -+.-> tornament_while -'+-> tornament_merger -|
        #                        |'-> raw_extractor -'^                     '-> tornament_merger -|
        # END   <----------------'                    '-------------------------------------------'
        #
        builder = StateGraph(AgenticState)
        builder.add_node("raw_loader", raw_loader_node)
        builder.add_node("raw_extractor", raw_extractor_node)
        builder.add_node("tournament_while", tournament_while_node)
        builder.add_node("tournament_merger", tournament_merger_node)
        builder.add_node("raw_increment", raw_increment_node)

        builder.add_edge(START, "raw_loader")
        builder.add_conditional_edges("raw_loader", raw_orchestrator_node,{"end": END, "raw_extractor": "raw_extractor"})
        builder.add_edge("raw_extractor", "tournament_while")
        builder.add_conditional_edges("tournament_while", tournament_orchestrator_node,{ "break": "raw_increment", "tournament_merger": "tournament_merger" })
        builder.add_edge("tournament_merger", "tournament_while")
        builder.add_edge("raw_increment", "raw_loader")

        graph = builder.compile()

        # Launch agentic scrapper
        logger.info(f"Launching agent execution...")
        final_state = await graph.ainvoke({
            "idir"         : args.idir,
            "odir"         : args.odir,
            "raw_fnames"   : raw_html_files,
            "raw_html_id"  : 0,
            "raw_html"     : "",
            "drafts"       : {},
            "drafts_count" : 0,
        })
        logger.info(f"\n==================== AGENT EXECUTION COMPLETE ====================")
    else                   :
        logger.info(f"\n==================== NO INPUT FILES FOUND ====================")
    return 0

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Gold Standard Dataset Pipeline Generator.")
    parser.add_argument("--idir", '-i', type=str, default="../html/test/", dest="idir", help="Folder containing source HTML variants")
    parser.add_argument("--odir", '-o', type=str, default="../html/test/", dest="odir", help="Folder to save pristine consensus Markdown outputs")

    parsed_args = parser.parse_args()

    # Pass the namespace to main, execute the event loop, and hand off exit integer to sys.exit()
    sys.exit(asyncio.run(main(parsed_args)))

