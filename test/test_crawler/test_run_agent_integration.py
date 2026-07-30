"""
Integration test: run_agent.py in DEBUG mode.

Real browser + mock back-ends on a live finance site.
deepness_max=2  retries_max=1  pages_max=2  html ⇒ test/html/

Asserts >=2 story archives (.html.gz) are saved.

    pytest test/test_crawler/test_run_agent_integration.py -v -s
"""

import datetime
import glob as glob_mod
import os
import shutil
import sys

# path wiring
_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
_CRW = os.path.join(_SRC, 'crawler')
for _p in (_SRC, _CRW):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest
from run_agent import run_stealth_graph, logger

# constants
_HTML_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'html'))
_SEED = "https://ca.finance.yahoo.com/"


def _today_dir() -> str:
    return os.path.join(_HTML_DIR, datetime.datetime.now().strftime('%Y-%m-%d'))


def _clean_today() -> None:
    for f in glob_mod.glob(os.path.join(_today_dir(), '*.html.gz')):
        os.remove(f)


def _archives() -> list[str]:
    return sorted(
        os.path.basename(p)
        for p in glob_mod.glob(os.path.join(_today_dir(), '*.html.gz'))
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_debug_agent_crawl_two_stories():
    os.environ['DEBUG'] = '1'
    _clean_today()

    error = await run_stealth_graph(
        url_string=_SEED,
        deepness_max=2,
        retries_max=1,
        pages_max=2,
        html_folder_name=_HTML_DIR,
    )

    assert error == 0, f"run_stealth_graph returned code {error}"

    files = _archives()
    td = _today_dir()
    for nm in files:
        print(f'  ✓  {nm}')

    assert len(files) >= 2, (
        f"Expected >=2 story archives in {td}, found {len(files)}"
    )
