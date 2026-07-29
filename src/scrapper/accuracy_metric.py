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
import numpy as np
import gzip
from gold_parser import BASE_URL, BASE_MODEL_NAME, BASE_API_KEY
#from langchain_openai import OpenAI
from openai import OpenAI



LLM_RAW  = OpenAI(base_url=BASE_URL + "/v1", api_key=BASE_API_KEY)

# Evaluate the draft
async def compute_perplexity_score(source : str, draft : str) -> float :
    # Compute perplexity
    llm_response = LLM_RAW.completions.create(
        model=BASE_MODEL_NAME,
        prompt=draft,
        max_tokens=0,
        echo=True,
        logprobs=5
    )
    token_logprobs = llm_response.choices[0].logprobs.token_logprobs
    valid_logprobs = [lp for lp in token_logprobs if lp is not None]
    perplexity = -sum(valid_logprobs) / len(valid_logprobs)
    return perplexity

async def main(args: argparse.Namespace) -> int :
    # Get scores
    scores = []
    for fname in os.listdir(args.idir) :
        if fname.endswith(".gz") :
            full_ipath = os.path.join(args.idir, fname)
            full_opath = os.path.join(args.odir, fname[:-len(".gz")] + ".txt")
            try :
                if os.path.isfile(full_ipath) and os.path.isfile(full_opath) :
                    with gzip.open(full_ipath, "rt", encoding="utf-8") as f :
                        raw_html = f.read()
                    with open(full_opath, "r", encoding="utf-8") as f :
                        draft = f.read()
            except Exception as e :
                logger.warning(f"Failed to read file {full_ipath}: {e}. Skipping to next.")
            score = await compute_perplexity_score(raw_html, draft)
            scores.append(score)
    # Show score
    logger.info(f"Scores: {scores}")
    return 0

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Gold Standard Dataset Pipeline Generator.")
    parser.add_argument("--idir", '-i', type=str, default="../html/test/", dest="idir", help="Folder containing source HTML variants")
    parser.add_argument("--odir", '-o', type=str, default="../html/test/", dest="odir", help="Folder to save pristine consensus Markdown outputs")

    parsed_args = parser.parse_args()

    # Pass the namespace to main, execute the event loop, and hand off exit integer to sys.exit()
    sys.exit(asyncio.run(main(parsed_args)))
