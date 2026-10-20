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

# =====================================================================
# Generic agentic processing harness (LangGraph)
# ---------------------------------------------------------------------
# A task-agnostic, self-correcting processing pipeline:
#
#             .----------------------------------------.
#             |        fail (budget left)              |
#             V                                        |
#     START -+=>  PROCESS --> DET_AUDIT --> LLM_AUDIT =+-> END
#
#     a "pass" verdict or a budget exhaustion exit at LLM_AUDIT -> END;
#     the + is the pipeline's single entry edge:
#     a retry re-enters through the same edge a fresh document takes
#     from START.
#
# The graph represents a sequence of increasingly expensive operations:
# the candidate output (PROCESS - one high-effort pass that does the
# task's coupled work, e.g. localise + extract), a free objective
# validation (DETERMINISTIC_AUDIT, never an LLM) and a semantic
# judgement (LLM_AUDIT) that is the graph's TERMINAL verdict. The
# deterministic audit is EVIDENCE for the LLM audit (it rides into the
# audit prompt) and its verdict AGREEMENT-CHECKS the LLM verdict
# deterministically (det fail -> llm fail, enforced harness-side - the
# LLM never sees the rule): acceptance is the LLM audit's "pass", which
# therefore implies a deterministic "pass" (or that the audit is
# unwired). The deterministic audit never short-circuits execution:
# whenever there is content to audit, both audits run (a process error
# is skipped by both - nothing to audit, no LLM audit call).
#
# The audit verdicts are BOUNDED BLOCKS. The LLM auditor grades the
# task's semantic metrics on the 3-valued scale and reports its
# observations as three text blocks + the one-word verdict (the
# contract's generation order: fail_stuff, none_stuff, pass_stuff,
# verdict - the verdict LAST, so it can only follow from the
# observations): each block is truncated to LLM_AUDIT_BLOCK_MAX_CHARS
# by the harness (the LLM is not trusted to keep its limit). The
# rendered deterministic audit block is asserted to stay under
# DET_AUDIT_BLOCK_MAX_CHARS at its single render site (the LLM
# audit's feedback compilation - the only place the block rides; the
# det audit is code - an exact bound, a bloated task implementation
# fails loud).
#
# The topology is IDENTICAL for every application (news article
# extraction, paper cleaning, section extraction, metadata extraction,
# RAG preprocessing, summarization verification, entity extraction, ...):
# the graph contains NO task-specific prompts and NO task-specific
# validation. Task behaviour is fixed at build time by the build_graph()
# parameters (the compiled graph is dedicated to one task):
#   1. a process prompt builder     (candidate output)
#   2. an audit prompt builder      (semantic verification)
#   3. a deterministic audit implementation (objective validation, no LLM;
#      BINARY - pass/fail, enforced; optional - None = no-op pass-through,
#      acceptance rests on the LLM audit alone)
# Run-level dependencies (the LLM client, the attempt budget) stay
# flexible: they ride the per-invocation context
# (runtime.context["llm_base"] / runtime.context["retries_max"]), so one
# compiled graph can be invoked with different LLM clients / budgets.
#
# State design: the state carries ONLY serializable per-document data -
# immutable inputs (document, document_name) and APPEND-ONLY histories
# (one entry appended per node per pass). Every state update is additive,
# which keeps the state compatible with LangGraph persistence, LangSmith
# time-travel and checkpointing. The current attempt number is always
# attempt = len(process_history); the latest result is always
# process_history[-1]. Only history[-1] of each history ever reaches the
# LLM context (the prompt builders receive single entries, never lists),
# and the correction context of a retry is the COMPILED FEEDBACK of the
# previous attempt - the concatenation of bounded blocks, bounded by
# construction - so the LLM context stays bounded no matter how many
# attempts ran. The accepted result (when the cycle ends on a pass) is
# process_history[-1] together with the terminal verdict's "pass".
#
# The retry decision belongs to the LLM_AUDIT conditional edge (the
# graph's single branching point): there is no retry loop inside any
# node. The only other in-node conditional is the single one-shot rerun
# of an unparseable LLM answer (a statement, not a loop).
#
# A process error (no usable content at all: an unreadable/empty process
# output after the one rerun) is a plain fail verdict that the audits
# skip: nothing is audited (no LLM audit call is spent) and the edge
# routes it per the attempt budget like any other fail. Everything else
# (fail grades, an unreadable AUDITOR verdict) is the same normal fail:
# the compiled feedback of the audits is carried into the next attempt's
# process prompt.
#
# On a non-pass verdict the LLM_AUDIT node also appends the COMPILED
# FEEDBACK to feedback_history: the verdict, the deterministic block
# (when the audit is wired) and the LLM's fail/none/pass observation
# blocks (each bounded) - what went wrong, what was good, what could
# not be graded. That new history[-1] member is the correction context
# the next attempt's prompt reads.
# =====================================================================

from functools import partial
import json
import operator
import re
from typing import ( Annotated, Any, Callable, Dict, List, Optional,
                     Tuple, TypedDict )

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime


# The grading scale of the audits: no scores, no probabilities, no
# confidence percentages. The LLM audit's verdict is one word of this
# scale ("none" = the auditor's epistemic uncertainty); the
# deterministic audit is BINARY (pass/fail - enforced in
# det_audit_node): a deterministic checker can never be "unsure", and
# its verdict agreement-checks the LLM verdict (det fail -> llm fail).
GRADES = ( "pass", "fail", "none" )

# The bounded-block budget of the audit verdicts. The LLM auditor's
# three observation blocks (fail/none/pass) are each truncated to
# LLM_AUDIT_BLOCK_MAX_CHARS chars - 2048 TOKENS at ~4 chars per token,
# no tokenizer call (the prompt asks the LLM for the same budget; the
# harness enforces it). The rendered deterministic audit block is
# asserted to stay under DET_AUDIT_BLOCK_MAX_CHARS at its single
# render site (the det audit is code, so the bound is exact). The
# compiled feedback is the
# concatenation of these bounded blocks, so the correction context is
# bounded by construction.
LLM_AUDIT_BLOCK_MAX_CHARS = 2048 * 4
DET_AUDIT_BLOCK_MAX_CHARS = 2048

# The no-op pass-through verdict of an unwired deterministic audit
# (det_audit=None at build time): the det_audit node emits it WITHOUT
# calling the task's callable, and the feedback compilation recognizes
# it by this marker (no deterministic block is added to the compiled
# feedback).
_DET_UNWIRED_ISSUE = "no deterministic audit wired for this task - skipped"


class AgenticState(TypedDict) :
    """
    The state of the generic processing harness. One ainvoke processes ONE
    document; the driver feeds one document per cycle.

    The state carries ONLY per-document data: every field is (de)serializable
    on purpose so a checkpointer (crash-resumable processing, LangSmith
    time-travel) can be added later without a state refactor. Run-level
    dependencies are NOT state: the task's routines are fixed at build time
    (build_graph parameters, bound to the nodes with functools.partial);
    the LLM client and the attempt budget are constant for the whole run
    and are injected per invocation via the plain-dict context of
    graph.ainvoke(context=...), reaching the nodes as
    runtime.context["llm_base"] / runtime.context["retries_max"].

    Attributes (immutable inputs):
        document (str): the full text of the document to process
            (the driver guarantees it is non-empty).
        document_name (str): the name of the document currently being
            processed; the nodes use it in their log lines.

    Attributes (append-only histories - one entry appended per node per
    pass, never rewritten):
        process_history: the candidate output of each attempt
            (task-specific fields; {"error"} on a no-usable-output
            failure - the audits skip such entries).
        det_audit_history: the objective verdict of each attempt:
            {"overall", "metrics", "issues", "feedback"} (BINARY:
            pass/fail; a no-op pass-through entry when the audit is
            unwired (None)).
        llm_audit_history: the semantic verdict of each attempt - the
            graph's TERMINAL verdict, agreement-checked by the
            deterministic verdict: {"verdict", "fail_stuff",
            "none_stuff", "pass_stuff"} (verdict = one word of the
            scale; the stuff fields are the auditor's grouped
            observations - one line per metric - each truncated to
            LLM_AUDIT_BLOCK_MAX_CHARS).
        feedback_history: the correction channel - one COMPILED
            FEEDBACK block per non-pass attempt (the verdict, the
            deterministic block when wired, the LLM's fail/none/pass
            blocks - each bounded).

    The current attempt number is always len(process_history); the latest
    result is always process_history[-1]. The accepted result (when the
    cycle ends on a pass) is process_history[-1] together with
    llm_audit_history[-1]["verdict"] == "pass".
    """
    document : str
    document_name : str
    process_history : Annotated[List[Dict[str, Any]], operator.add]
    det_audit_history : Annotated[List[Dict[str, Any]], operator.add]
    llm_audit_history : Annotated[List[Dict[str, Any]], operator.add]
    feedback_history : Annotated[List[str], operator.add]


async def _json_from_llm(llm_base : ChatOpenAI, system_prompt : str, user_prompt : str, tag : str) -> Optional[Dict[str, Any]] :
    """
    Call the LLM with a system + user prompt and recover the JSON object of its
    answer.

    Tolerates the usual structured-output failure modes: `content` returned as
    a list of blocks, a markdown code fence around the JSON, and prose wrapping
    the JSON. Returns the parsed dict, or None when no JSON object can be
    recovered (the caller decides the one-time rerun policy).
    """
    try :
        response = await llm_base.ainvoke([ SystemMessage(content=system_prompt), HumanMessage(content=user_prompt) ])
    except Exception as e :
        logger.error(f"LLM {tag} call failed: {e}")
        return None
    raw = response.content if ( response is not None and getattr(response, "content", None) is not None ) else ""
    if isinstance(raw, list) :   # content may be a list of blocks
        raw = "".join( part.get("text", "") if isinstance(part, dict) else str(part) for part in raw )
    if not isinstance(raw, str) :
        raw = str(raw)
    text = raw.strip()
    if not text :
        logger.error(f"LLM {tag} returned an empty response.")
        return None
    try :
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError) :
        pass
    fenced = re.match(r"\s*```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced :
        try :
            obj = json.loads(fenced.group(1))
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, TypeError) :
            pass
    # Last resort: scan for the first balanced {...} block inside the text.
    for brace in re.finditer(r"\{", text) :
        depth, in_str, esc = 0, False, False
        for i in range(brace.start(), len(text)) :
            ch = text[i]
            if in_str :
                if esc : esc = False
                elif ch == "\\" : esc = True
                elif ch == '"' : in_str = False
            elif ch == '"' : in_str = True
            elif ch == "{" : depth += 1
            elif ch == "}" :
                depth -= 1
                if depth == 0 :
                    try :
                        obj = json.loads(text[brace.start():i+1])
                        return obj if isinstance(obj, dict) else None
                    except (json.JSONDecodeError, TypeError) :
                        break
    logger.error(f"LLM {tag} response contained no parseable JSON object.")
    return None


# =====================================================================
# The graph nodes
# ---------------------------------------------------------------------
# Every node is a plain function: the task's routine is bound at graph
# assembly with functools.partial (a build_graph parameter), and the LLM
# client comes from the per-invocation context. Every node reads
# the immutable inputs + the LATEST entry of the needed histories (never
# whole lists), does ONE stage of the pipeline, and appends exactly one
# entry to its own history. Retries are the LLM_AUDIT conditional edge,
# not node loops.
# =====================================================================


async def process_node(state : AgenticState, runtime : Runtime, *,
                       build_process_prompt : Callable[[str, str, str], Tuple[str, str]]) -> Dict[str, Any] :
    """
    PROCESS: generate the candidate output (one LLM call; the task's
    process prompt builder owns the prompt). The single high-effort pass
    does the task's coupled work (e.g. localise + extract).
    Flow:
      1. read the context + the latest feedback (empty on the first attempt)
      2. build the prompt via the task's builder
      3. LLM call + one in-place rerun on an unparseable answer
      4. record the candidate (an error entry when no usable output
         exists at all - the audits skip it, the edge routes per budget)
    """
    document = state["document"]
    name = state["document_name"]
    llm_base = runtime.context["llm_base"]
    attempt = len(state["process_history"]) + 1

    # Step 1: the correction channel (the compiled feedback of the
    # previous attempt; empty on the first attempt).
    feedback = state["feedback_history"][-1] if state["feedback_history"] else ""

    # Step 2: the task owns the prompt (injected at build time).
    (system, user) = build_process_prompt(document, name, feedback)

    # Step 3: one LLM call + one in-place rerun (identical inputs) before
    # the attempt is counted as failed.
    result = await _json_from_llm(llm_base, system, user, "process")
    if result is None :
        logger.warning(f"{name} attempt {attempt}: LLM processing failed "
                       f"(no usable process answer) - one rerun")
        result = await _json_from_llm(llm_base, system, user, "process (rerun)")
        if result is None :
            logger.error(f"{name} attempt {attempt}: LLM processing failed "
                         f"(no usable process answer after the rerun)")

    # Step 4: record the entry. No usable output at all (unparseable or an
    # output without any non-empty field) is a process error: the audits
    # skip it (no LLM audit call) and the edge routes it per the budget.
    if result is None or not any(v not in (None, "", [], {}) for v in result.values()) :
        entry = { "error" : "process output unreadable or empty after rerun" }
        logger.error(f"{name} attempt {attempt}: PROCESS produced no usable result - "
                     f"the audits skip it (no LLM audit call)")
        return { "process_history" : [entry] }
    logger.info(f"{name} attempt {attempt}: PROCESS ok ({len(result)} field(s))")
    return { "process_history" : [result] }


async def det_audit_node(state : AgenticState, *,
                         det_audit : Optional[Callable[[str, str, Dict[str, Any]], Dict[str, Any]]]) -> Dict[str, Any] :
    """
    DETERMINISTIC AUDIT: objective validation (the task's deterministic
    audit implementation; NEVER an LLM). It is BINARY (pass/fail): a
    deterministic checker can never be "unsure" - "none" is the LLM
    auditor's epistemic grade and a "none" verdict is a contract
    violation (the check below fails loud). Its verdict is ground-truth
    EVIDENCE for the LLM audit (it rides into the audit prompt) and it
    AGREEMENT-CHECKS the LLM verdict (det fail -> llm fail, enforced
    harness-side). It rides into the compiled feedback of the next
    attempt, rendered at that single consumption site (the LLM audit
    node), where it is asserted <= DET_AUDIT_BLOCK_MAX_CHARS. The
    audit is OPTIONAL at build time: when it is
    unwired (None) the node emits a no-op pass-through verdict WITHOUT
    calling the task's callable, so the graph runs unchanged and
    acceptance rests on the LLM audit alone. It never short-circuits
    execution (it is free).
    Flow:
      1. an upstream process error yields a fail verdict without calling
         the task's audit (nothing to audit - no LLM was ever involved)
      2. an unwired audit (None) yields a pass-through pass verdict
         (no objective validation is wired - the LLM audit decides)
      3. the task computes the verdict on the LATEST process entry
      4. the verdict schema is normalized fail-safe (the task owns the
         grades; a broken verdict fails the attempt) and the binary
         contract is enforced (a "none" verdict fails loud)
    """
    document = state["document"]
    name = state["document_name"]
    attempt = len(state["det_audit_history"]) + 1

    # Step 1: an upstream process error - nothing to audit.
    process = state["process_history"][-1]
    if process.get("error") :
        verdict = { "overall" : "fail", "metrics" : {},
                    "issues" : [ process["error"] ], "feedback" : "" }
        logger.info(f"{name} attempt {attempt}: DETERMINISTIC AUDIT: process error upstream - skipped")
        return { "det_audit_history" : [verdict] }

    # Step 2: the audit is optional - an unwired audit (None) is a no-op
    # pass-through (acceptance then rests on the LLM audit alone; the
    # task's callable is never invoked).
    if det_audit is None :
        verdict = { "overall" : "pass", "metrics" : {},
                    "issues" : [ _DET_UNWIRED_ISSUE ],
                    "feedback" : "" }
        logger.info(f"{name} attempt {attempt}: DETERMINISTIC AUDIT skipped (not wired)")
        return { "det_audit_history" : [verdict] }

    # Step 3: the task owns the objective validation (no LLM involved).
    verdict = det_audit(document, name, process)

    # Step 4: fail-safe schema normalization + the binary contract. The
    # task's grade is read and checked exactly once (a broken verdict
    # fails the attempt instead of crashing the graph):
    overall = verdict.get("overall") if isinstance(verdict, dict) else None
    if overall not in ( "pass", "fail" ) :
        # A "none" grade is the LLM auditor's epistemic grade - the
        # binary contract says the deterministic audit can never be
        # "unsure" -> fail loud.
        if overall == "none" :
            raise AssertionError(
                f"{name}: the deterministic audit must return a binary verdict (pass/fail), got 'none'")
        # An ambiguous breakage (not a dict, a missing or an off-scale
        # grade) -> normalize to a well-formed fail verdict.
        raw = verdict if isinstance(verdict, dict) else {}
        verdict = { "overall" : "fail",
                    "metrics" : raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {},
                    "issues" : [ "deterministic audit returned no valid verdict" ],
                    "feedback" : str(raw) }
    verdict.setdefault("metrics", {})
    verdict.setdefault("issues", [])
    verdict.setdefault("feedback", "")
    fails = [ k for k, v in verdict["metrics"].items() if v == "fail" ] \
        if isinstance(verdict["metrics"], dict) else []
    logger.info(f"{name} attempt {attempt}: DETERMINISTIC AUDIT {verdict['overall']}"
                + (f" (fail: {', '.join(fails)})" if fails else ""))
    return { "det_audit_history" : [verdict] }


async def llm_audit_node(state : AgenticState, runtime : Runtime, *,
                         build_audit_prompt : Callable[[str, str, Dict[str, Any],
                                                        Dict[str, Any]],
                                                        Tuple[str, str]]) -> Dict[str, Any] :
    """
    LLM AUDIT: semantic verification (one LLM call; the task's audit
    prompt builder owns the prompt) and the graph's TERMINAL verdict.
    The auditor grades the task's semantic metrics on the 3-valued scale
    and reports its observations as bounded text blocks - the contract
    (in generation order): fail_stuff, none_stuff, pass_stuff, verdict -
    the verdict LAST, so it can only follow from the observations; each
    block is truncated to LLM_AUDIT_BLOCK_MAX_CHARS here (the LLM is
    not trusted to keep its limit). The deterministic verdict
    AGREEMENT-CHECKS the LLM verdict (det fail -> llm fail, enforced
    harness-side - the LLM never sees the rule), so this verdict's
    "pass" implies a deterministic "pass" (or that the deterministic
    audit is unwired). The outgoing conditional edge routes on the
    recorded verdict (the graph's single branching point).
    Flow:
      1. an upstream process error yields a fail verdict WITHOUT an LLM
         call (nothing to review); the compiled feedback still rides
         into feedback_history and the edge routes it per the budget
      2. the prompt carries the LATEST process entry and the
         deterministic audit as objective evidence
      3. LLM call + one in-place rerun on an unparseable answer
      4. the verdict schema is normalized fail-safe: the verdict must be
         one word of the scale (anything else is forced to "fail"); the
         three stuff fields are stringified and truncated to
         LLM_AUDIT_BLOCK_MAX_CHARS
      5. the agreement check: a deterministic fail verdict dominates the
         LLM verdict (det fail -> llm fail) - the deterministic detail
         rides into fail_stuff so the correction channel shows the
         override
      6. on a non-pass verdict the COMPILED FEEDBACK (the verdict, the
         deterministic block when wired, the LLM's fail/none/pass blocks
         - each bounded) is appended to feedback_history - the
         correction context the next attempt's process prompt reads; the
         conditional edge retries while the attempt budget allows and
         ends on a pass verdict or a budget exhaustion
    """
    document = state["document"]
    name = state["document_name"]
    llm_base = runtime.context["llm_base"]
    attempt = len(state["llm_audit_history"]) + 1

    # Step 1: a process error leaves nothing to review (no LLM call). The
    # terminal verdict is a plain fail and the compiled feedback still
    # rides into feedback_history (every non-pass attempt feeds the
    # correction channel); the edge routes it per the budget like any
    # other fail.
    process = state["process_history"][-1]
    if process.get("error") :
        verdict = { "verdict" : "fail", "pass_stuff" : "",
                    "fail_stuff" : str(process["error"]), "none_stuff" : "" }
        logger.info(f"{name} attempt {attempt}: LLM AUDIT skipped (process error upstream)")
        retries_max = runtime.context["retries_max"]
        if attempt >= retries_max :
            logger.error(f"{name}: LLM_AUDIT -> END (attempt budget of {retries_max} exhausted "
                         f"without an accepted result)")
        else :
            logger.info(f"{name}: LLM_AUDIT -> PROCESS (attempt {attempt + 1} of {retries_max})")
        summary = ( f"{attempt} attempt(s) failed so far; latest (attempt {attempt}):"
                    f"\n  verdict: fail"
                    f"\n  process error: {process['error']}"
                    f"\n  re-run the extraction with a clean, parseable answer." )
        return { "llm_audit_history" : [verdict],
                 "feedback_history" : [summary] }

    # Step 2: the task owns the prompt; the deterministic audit rides in as
    # objective evidence (the auditor does not re-derive its numbers).
    det_audit = state["det_audit_history"][-1]
    (system, user) = build_audit_prompt(document, name, process, det_audit)

    # Step 3: one LLM call + one in-place rerun (identical inputs) before
    # the attempt is counted as failed.
    result = await _json_from_llm(llm_base, system, user, "llm_audit")
    if result is None :
        logger.warning(f"{name} attempt {attempt}: LLM processing failed "
                       f"(no usable audit answer) - one rerun")
        result = await _json_from_llm(llm_base, system, user, "llm_audit (rerun)")
        if result is None :
            logger.error(f"{name} attempt {attempt}: LLM processing failed "
                         f"(no usable audit answer after the rerun)")

    # Step 4: fail-safe schema normalization (a broken verdict fails the
    # attempt instead of crashing the graph). The auditor's contract:
    # {"verdict", "fail_stuff", "none_stuff", "pass_stuff"} - the
    # verdict is one word of the scale (anything else is forced to
    # "fail"); the stuff fields are the bounded observation blocks
    # (stringified + truncated to LLM_AUDIT_BLOCK_MAX_CHARS - the LLM
    # is not trusted to keep its limit).
    if result is None :
        verdict = { "verdict" : "fail", "pass_stuff" : "",
                    "fail_stuff" : "the audit of the previous attempt could not be read; "
                                   "re-run the extraction",
                    "none_stuff" : "" }
        logger.error(f"{name} attempt {attempt}: LLM AUDIT verdict unreadable (normal failure)")
    else :
        verdict_word = str(result.get("verdict", "")).strip()
        if verdict_word not in GRADES :
            verdict_word = "fail"
            logger.error(f"{name} attempt {attempt}: LLM AUDIT verdict off the scale "
                         f"(forced fail)")
        verdict = { "verdict" : verdict_word }
        for key in ( "fail_stuff", "none_stuff", "pass_stuff" ) :
            text = str(result.get(key, "")).strip()
            verdict[key] = text[:LLM_AUDIT_BLOCK_MAX_CHARS] if len(text) > LLM_AUDIT_BLOCK_MAX_CHARS else text 

    # Step 5: the agreement check (the floor) - the deterministic verdict
    # is ground truth the LLM verdict cannot outrank: when the
    # deterministic audit fails, the LLM verdict is forced to "fail"
    # (a deterministic checker is never "unsure"). Enforced harness-side
    # (the LLM never sees this rule); the deterministic detail rides
    # into fail_stuff so the correction channel shows the override.
    if det_audit.get("overall") == "fail" and verdict["verdict"] != "fail" :
        logger.warning(f"{name} attempt {attempt}: LLM AUDIT verdict overridden by the "
                       f"deterministic audit (det fail -> llm fail)")
        verdict["verdict"] = "fail"
        note_source = str(det_audit.get("feedback", "")).strip() \
            or "; ".join( str(issue) for issue in (det_audit.get("issues") or [])[:2] )
        verdict["fail_stuff"] = ( verdict["fail_stuff"]
                                  + "\nthe deterministic audit verdict is fail - the LLM "
                                    "verdict is overridden: " + note_source ).strip()
        if len(verdict["fail_stuff"]) > LLM_AUDIT_BLOCK_MAX_CHARS :
            verdict["fail_stuff"] = verdict["fail_stuff"][:LLM_AUDIT_BLOCK_MAX_CHARS]
    logger.info(f"{name} attempt {attempt}: LLM AUDIT {verdict['verdict']}")

    # Step 6: the correction channel - on a non-pass verdict the COMPILED
    # FEEDBACK (the verdict, the deterministic block when wired, the
    # LLM's fail/none/pass blocks - each bounded: det asserted <=
    # DET_AUDIT_BLOCK_MAX_CHARS, LLM blocks truncated to
    # LLM_AUDIT_BLOCK_MAX_CHARS) is appended to feedback_history - that
    # new history[-1] member is what the next attempt's prompt reads. The
    # conditional edge then retries while the attempt budget allows and
    # ends once it is exhausted.
    if verdict["verdict"] != "pass" :
        lines = [ f"{attempt} attempt(s) failed so far; latest (attempt {attempt}):",
                  f"  verdict: {verdict['verdict']}" ]
        if det_audit.get("issues") != [ _DET_UNWIRED_ISSUE ] :
            # The det block is rendered ONCE (here, its single
            # consumption site) and asserted bounded before it rides
            # into the correction channel (the det audit is code - a
            # bloated task implementation fails loud).
            det_lines = [ f"  deterministic audit: {det_audit.get('overall')}" ]
            for issue in det_audit.get("issues") or [] :
                det_lines.append(f"  - {issue}")
            det_feedback = str(det_audit.get("feedback", "")).strip()
            if det_feedback :
                det_lines.append(f"  - feedback: {det_feedback}")
            det_block = "\n".join(det_lines)
            assert len(det_block) <= DET_AUDIT_BLOCK_MAX_CHARS, \
                f"{name}: the deterministic audit block exceeds {DET_AUDIT_BLOCK_MAX_CHARS} " \
                f"chars (got {len(det_block)})"
            lines += det_lines
        lines.append("  llm audit:")
        for label, key in ( ("fail", "fail_stuff"), ("none", "none_stuff"),
                            ("pass", "pass_stuff") ) :
            if verdict[key] :
                lines.append(f"    {label}: {verdict[key]}")
        if not any(verdict[key] for key in ( "fail_stuff", "none_stuff", "pass_stuff" )) :
            lines.append("    (no audit detail reported)")
        summary = "\n".join(lines)
        retries_max = runtime.context["retries_max"]
        if attempt >= retries_max :
            logger.error(f"{name}: LLM_AUDIT -> END (attempt budget of {retries_max} exhausted "
                         f"without an accepted result)")
        else :
            logger.info(f"{name}: LLM_AUDIT -> PROCESS (attempt {attempt + 1} of {retries_max})")
        return { "llm_audit_history" : [verdict],
                 "feedback_history" : [summary] }
    logger.info(f"{name} attempt {attempt}: LLM AUDIT PASSED (the terminal verdict)")
    return { "llm_audit_history" : [verdict] }


def build_graph(*,
                build_process_prompt : Callable[[str, str, str], Tuple[str, str]],
                build_audit_prompt : Callable[[str, str, Dict[str, Any],
                                               Dict[str, Any]], Tuple[str, str]],
                det_audit : Optional[Callable[[str, str, Dict[str, Any]], Dict[str, Any]]] = None
                ) :
    """
    Compile the task-specific processing graph (the TOPOLOGY is identical
    for every task; the task behaviour is fixed at build time through the
    three routine parameters):

            .---------------------------------------.
            |        fail (budget left)             |
            V                                       |
     START -+=>  PROCESS -> DET_AUDIT -> LLM_AUDIT =+-> END

    Parameters (the task interface - each callable's contract):
        build_process_prompt(document, document_name, feedback)
            -> (system_prompt, user_prompt)
        build_audit_prompt(document, document_name, process_result,
                           deterministic_audit)
            -> (system_prompt, user_prompt)
        det_audit(document, document_name, process_result)
            -> the verdict dict {"overall", "metrics", "issues", "feedback"}
            (BINARY: pass/fail - enforced; BOUNDED: the rendered block
            <= DET_AUDIT_BLOCK_MAX_CHARS - asserted; OPTIONAL: None =
            the audit is unwired; the det_audit node then emits a no-op
            pass-through verdict without calling the task's callable,
            so acceptance rests on the LLM audit alone - the graph
            topology is unchanged)

    The LLM_AUDIT verdict is the graph's terminal verdict: it carries
    the deterministic verdict as evidence and is agreement-checked by
    it (det fail -> llm fail), so its "pass" implies a deterministic
    "pass". Run-level dependencies stay flexible: they ride the
    per-invocation context -
    graph.ainvoke(input, context={ "llm_base" : <ChatOpenAI>,
    "retries_max" : <int> }).

    The only outgoing edge from LLM_AUDIT is an explicit conditional edge
    (the graph's single branching point): one field check on the terminal
    verdict - END when the verdict is "pass" or the attempt budget is
    exhausted, PROCESS otherwise. The retry re-enters through the
    pipeline's single entry edge (the same edge a fresh document takes
    from START).
    """
    builder = StateGraph(AgenticState)
    builder.add_node("process", partial(process_node, build_process_prompt=build_process_prompt))
    # The det_audit node is ALWAYS registered (the static edges reference
    # it): the optionality of the audit lives inside the node, which emits
    # a no-op pass-through verdict when the audit is unwired (None).
    builder.add_node("det_audit", partial(det_audit_node, det_audit=det_audit))
    builder.add_node("llm_audit", partial(llm_audit_node, build_audit_prompt=build_audit_prompt))
    builder.add_edge(START, "process")
    builder.add_edge("process", "det_audit")
    builder.add_edge("det_audit", "llm_audit")
    # The graph's single branching point: one field check on the terminal
    # LLM_AUDIT verdict ("pass" = accept) + the attempt budget.
    builder.add_conditional_edges("llm_audit",
        lambda state, runtime : END if (
            state["llm_audit_history"][-1].get("verdict") == "pass"
            or len(state["process_history"]) >= runtime.context["retries_max"]
        ) else "process",
        { END : END, "process" : "process" })
    return builder.compile()
