#!/usr/bin/env python3
"""
PROJECT: Factor-Lake Portfolio Analysis
MODULE: scripts/load_test_sessions.py
PURPOSE: Runs N simultaneous browser sessions through a full backtest to check the app under load.

Each simulated user opens the WRDS-only app, sets its WRDS settings in the sidebar (a
rebalancing frequency and an initial AUM), selects a factor, clicks Run Portfolio Analysis
(the backtest runs in AWS Lambda; nothing is loaded in the app), opens Results and runs the
Top vs Bottom cohort comparison. The AUM differs per session and per script run, so every
backtest is computed rather than read from the Lambda result cache. The script exits with
status 1 if any session fails or shows an error.

Usage:
    pip install playwright && playwright install chromium
    python scripts/load_test_sessions.py                                   # live app, 5 users
    python scripts/load_test_sessions.py --url http://localhost:8501 --rounds 3
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.async_api import Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

DEFAULT_URL = "https://cornellfactorlake.streamlit.app/"
# WRDS factor names, as on the Analysis tab's factor buttons
FACTORS = ['12-1 Mo Momentum', 'ROE', 'ROA', 'Book/Price', '6-Mo Momentum']
FREQUENCIES = ['Quarterly', 'Monthly', 'Semi-annual']
READY_BUTTON = "Run Portfolio Analysis"
STEP_TIMEOUT_MS = 180_000


async def _app_root(page: Page) -> Any:
    """Streamlit Community Cloud serves the app inside an iframe; local runs do not."""
    wake: Any = page.get_by_role("button", name="get this app back up")
    try:
        await wake.wait_for(timeout=5_000)
    except PlaywrightTimeoutError:
        wake = None  # no sleep screen: the app is already awake
    if wake is not None:
        await wake.click()
    deadline = time.monotonic() + STEP_TIMEOUT_MS / 1000
    while time.monotonic() < deadline:
        frame_el = await page.query_selector('iframe[title="streamlitApp"]')
        root: Any = (await frame_el.content_frame()) if frame_el else page
        if root and await root.get_by_role("button", name=READY_BUTTON).count():
            return root
        await page.wait_for_timeout(1_000)
    raise TimeoutError(f"The app did not show the {READY_BUTTON} button")


async def _idle(root: Any) -> None:
    """Waits until Streamlit has finished the current script run."""
    await asyncio.sleep(0.5)
    await root.wait_for_function(
        "() => { const w = document.querySelector('[data-testid=stStatusWidget]');"
        " return !w || !w.innerText.includes('Running'); }",
        timeout=STEP_TIMEOUT_MS,
    )
    await asyncio.sleep(0.5)


async def _problems(root: Any) -> List[str]:
    texts = await root.locator('[data-testid=stAlertContentError], [data-testid=stException]').all_inner_texts()
    return [" ".join(t.split())[:300] for t in texts]


async def simulate_user(browser: Any, url: str, idx: int, round_no: int, aum: int,
                        screenshot_dir: Optional[Path]) -> Dict[str, Any]:
    """One user's full WRDS backtest. Returns timings in ms and any problems seen."""
    context = await browser.new_context(viewport={"width": 1400, "height": 900})
    page = await context.new_page()
    page_errors: List[str] = []
    page.on("pageerror", lambda e: page_errors.append(str(e)[:200]))
    factor = FACTORS[idx % len(FACTORS)]
    frequency = FREQUENCIES[idx % len(FREQUENCIES)]
    timings: Dict[str, int] = {}
    step = "open"
    start = time.monotonic()

    def lap(name: str, since: float) -> float:
        timings[name] = int((time.monotonic() - since) * 1000)
        return time.monotonic()

    try:
        await page.goto(url, timeout=STEP_TIMEOUT_MS)
        root = await _app_root(page)
        t = lap("open", start)

        step = "settings"
        sidebar = root.locator('[data-testid=stSidebar]')
        await sidebar.get_by_text(frequency, exact=True).click()
        await _idle(root)
        aum_input = sidebar.locator('input[aria-label="Initial AUM ($)"]')
        await aum_input.fill(str(aum))
        await aum_input.press("Enter")
        await _idle(root)
        # The factor buttons start with an icon, so match the button holding the factor's exact label
        await root.locator("button").filter(has=root.get_by_text(factor, exact=True)).first.click()
        await _idle(root)
        t = lap("settings", t)

        step = "run"
        await root.get_by_role("button", name=READY_BUTTON).click()
        await _idle(root)
        await root.get_by_text("Analysis complete").first.wait_for(timeout=STEP_TIMEOUT_MS)
        t = lap("run", t)

        step = "results"
        await root.get_by_role("tab", name="Results", exact=True).click()
        await _idle(root)
        metric = root.locator('[data-testid=stMetric]', has_text="Final Portfolio Value").first
        final_value = " ".join((await metric.inner_text()).split())
        t = lap("results", t)

        step = "cohort"
        await root.get_by_text("Run Cohort Comparison").click()
        await _idle(root)
        await root.get_by_role("button", name="Generate Comparison").click()
        await _idle(root)
        # The cohort summary table has rendered (its column header), not just its heading
        await root.get_by_text("Cohort Performance Summary").first.wait_for(timeout=STEP_TIMEOUT_MS)
        await root.get_by_text("Top Cohort", exact=True).first.wait_for(timeout=STEP_TIMEOUT_MS)
        lap("cohort", t)

        problems = await _problems(root)
        result = {"ok": not problems and not page_errors, "final_value": final_value}
    except Exception as e:
        try:
            problems = await _problems(page)
        except Exception as read_error:
            problems = [f"could not read the page: {str(read_error)[:200]}"]
        result = {"ok": False, "failed_at": step, "error": str(e).splitlines()[0][:300]}

    if screenshot_dir:
        suffix = "" if result["ok"] else "_FAIL"
        await page.screenshot(path=str(screenshot_dir / f"round{round_no}_user{idx}{suffix}.png"))
    await context.close()
    result.update({"user": idx, "factor": factor, "frequency": frequency, "aum": aum, "ms": timings,
                   "total_ms": int((time.monotonic() - start) * 1000),
                   "problems": problems, "page_errors": page_errors})
    return result


async def run(url: str, users: int, rounds: int, screenshot_dir: Optional[Path],
              executable_path: Optional[str]) -> bool:
    if screenshot_dir:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=executable_path)
        # A different AUM per script run, round and session, so no backtest is a cache hit
        base_aum = 10_000 + int(time.time()) % 100_000 * 10
        for round_no in range(1, rounds + 1):
            started = time.monotonic()
            # Every user starts at the same moment
            results = await asyncio.gather(*[
                simulate_user(browser, url, i, round_no, base_aum + 1_000_000 * round_no + i, screenshot_dir)
                for i in range(users)
            ])
            passed = sum(r["ok"] for r in results)
            print(f"ROUND {round_no}: {passed}/{users} sessions completed without errors "
                  f"in {time.monotonic() - started:.1f}s")
            for r in results:
                print("  " + json.dumps(r))
            all_ok = all_ok and passed == users
        await browser.close()
    return all_ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Simultaneous-session load test for the Factor-Lake app.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"App URL (default: {DEFAULT_URL})")
    parser.add_argument("--users", type=int, default=5, help="Simultaneous sessions per round (default: 5)")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds (default: 1)")
    parser.add_argument("--screenshots", type=Path, help="Folder for one screenshot per session")
    parser.add_argument("--chromium", help="Path to a Chromium binary (default: Playwright's own)")
    args = parser.parse_args(argv)
    ok = asyncio.run(run(args.url, args.users, args.rounds, args.screenshots, args.chromium))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
