"""
agent.py — Text Summarizer Agent (OpenAI)
=========================================
Single-agent workflow that accepts plain text, URL, or file input
and returns a structured summary using the OpenAI Chat Completions API.

Usage:
    python src/agent.py --text "Paste your text here" --style brief
    python src/agent.py --url https://example.com --style bullet
    python src/agent.py --file path/to/doc.pdf --style detailed

Author: Generated via text-summarizer-agent prompt guide
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

# ── Local imports ─────────────────────────────────────────────────────────────
# Assumes agent.py lives in src/ alongside tools.py and utils.py
sys.path.insert(0, str(Path(__file__).parent))
from tools import load_plain_text, load_from_url, load_from_file
from utils import chunk_text, count_tokens, clean_text, format_summary

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")          # flexible — override in .env
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "3000"))        # tokens per chunk
DEFAULT_STYLE: str = os.getenv("SUMMARY_STYLE", "brief")

VALID_STYLES = {"brief", "detailed", "bullet"}

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional document summarization agent. Your job is to read any \
text — from documents, web pages, or raw input — and produce accurate, structured summaries.

Rules:
- Never hallucinate or infer facts not present in the source text
- Preserve key names, numbers, dates, and decisions exactly as they appear
- Match the requested style exactly:
    BRIEF:    3-5 sentences, highest-level takeaway only
    DETAILED: structured paragraphs covering all major points
    BULLET:   5-10 bullet points, each a single clear fact or insight
- If the text is fragmented (chunked), synthesize all parts into one coherent summary
- Output only the summary — no preamble, no meta-commentary, no self-reference"""


# ── Core summarization logic ───────────────────────────────────────────────────

def build_user_prompt(text: str, style: str) -> str:
    """Construct the user-facing prompt with style instruction."""
    style_instruction = {
        "brief": (
            "Provide a BRIEF summary: 3-5 sentences capturing only the highest-level takeaway."
        ),
        "detailed": (
            "Provide a DETAILED summary: structured paragraphs covering all major themes, "
            "arguments, facts, and conclusions present in the text."
        ),
        "bullet": (
            "Provide a BULLET POINT summary: 5-10 concise bullet points, each expressing "
            "one distinct fact, finding, or key insight."
        ),
    }
    return (
        f"Summarize the following text.\n\n"
        f"Style: {style_instruction[style]}\n\n"
        f"--- TEXT START ---\n{text}\n--- TEXT END ---"
    )


def call_openai(client: OpenAI, user_prompt: str) -> str:
    """
    Send a single Chat Completions request and return the response text.

    Args:
        client:      Authenticated OpenAI client instance.
        user_prompt: Fully assembled user message string.

    Returns:
        The model's reply as a stripped string.

    Raises:
        RuntimeError: If the API returns no content.
    """
    log.info("Calling OpenAI model: %s  max_tokens: %d", MODEL_NAME, MAX_TOKENS)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=0.3,          # lower temp → more deterministic summaries
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty response. Check your API key and model name.")
    log.info(
        "Tokens used — prompt: %d  completion: %d  total: %d",
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.usage.total_tokens,
    )
    return content.strip()


def summarize_chunks(client: OpenAI, chunks: list[str], style: str) -> str:
    """
    Summarize each chunk individually, then combine into a final summary.

    Used automatically when the source text exceeds CHUNK_SIZE tokens.

    Args:
        client: Authenticated OpenAI client instance.
        chunks: List of text segments.
        style:  One of 'brief', 'detailed', 'bullet'.

    Returns:
        Final combined summary string.
    """
    log.info("Document chunked into %d parts — summarizing each...", len(chunks))
    partial_summaries: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        console.print(f"  [dim]Summarizing chunk {i}/{len(chunks)}...[/dim]")
        prompt = build_user_prompt(chunk, style="detailed")   # always detailed for chunks
        partial = call_openai(client, prompt)
        partial_summaries.append(partial)

    # Combine partial summaries into a final pass
    combined_text = "\n\n".join(partial_summaries)
    log.info("Combining %d partial summaries into final output...", len(partial_summaries))
    final_prompt = (
        f"The following are partial summaries of a long document. "
        f"Synthesize them into one coherent final summary.\n\n"
        + build_user_prompt(combined_text, style)
    )
    return call_openai(client, final_prompt)


def summarize(
    text: str | None = None,
    url:  str | None = None,
    file: str | None = None,
    style: str = DEFAULT_STYLE,
) -> str:
    """
    Main summarization entry point. Accepts one of three input types.

    Args:
        text:  Raw string of text to summarize.
        url:   Web URL to scrape and summarize.
        file:  Path to a .txt, .pdf, or .docx file.
        style: Summary style — 'brief' | 'detailed' | 'bullet'.

    Returns:
        The final summary as a string.

    Raises:
        ValueError: If no input is supplied or an invalid style is given.
        EnvironmentError: If OPENAI_API_KEY is missing.
    """
    # ── Validate inputs ────────────────────────────────────────────────────
    if not any([text, url, file]):
        raise ValueError("Provide at least one input: --text, --url, or --file.")

    if style not in VALID_STYLES:
        raise ValueError(f"Invalid style '{style}'. Choose from: {', '.join(VALID_STYLES)}")

    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    # ── Load source text ───────────────────────────────────────────────────
    client = OpenAI(api_key=OPENAI_API_KEY)

    if text:
        log.info("Input mode: plain text (%d chars)", len(text))
        raw = load_plain_text(text)
        source_label = "Plain text input"
    elif url:
        log.info("Input mode: URL — %s", url)
        console.print(f"[dim]Fetching content from {url}...[/dim]")
        raw = load_from_url(url)
        source_label = f"URL: {url}"
    else:
        log.info("Input mode: file — %s", file)
        console.print(f"[dim]Reading file: {file}...[/dim]")
        raw = load_from_file(file)
        source_label = f"File: {Path(file).name}"

    # ── Pre-process ────────────────────────────────────────────────────────
    cleaned = clean_text(raw)
    token_estimate = count_tokens(cleaned)
    log.info("Estimated tokens in source text: %d", token_estimate)

    # ── Summarize (single pass or chunked) ─────────────────────────────────
    console.print(Rule("[bold]Summarizer Agent[/bold]"))
    console.print(f"[dim]Source:[/dim]  {source_label}")
    console.print(f"[dim]Style:[/dim]   {style.upper()}")
    console.print(f"[dim]Model:[/dim]   {MODEL_NAME}")
    console.print(f"[dim]Tokens:[/dim]  ~{token_estimate:,} estimated\n")

    if token_estimate > CHUNK_SIZE:
        chunks = chunk_text(cleaned, CHUNK_SIZE)
        summary = summarize_chunks(client, chunks, style)
    else:
        prompt = build_user_prompt(cleaned, style)
        summary = call_openai(client, prompt)

    # ── Post-process & display ─────────────────────────────────────────────
    formatted = format_summary(summary, style)

    console.print(
        Panel(
            Text(formatted, style="default"),
            title=f"[bold]Summary — {style.upper()}[/bold]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )
    console.print(Rule())
    return formatted


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="text-summarizer-agent",
        description="Summarize text from plain input, a URL, or a file using OpenAI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/agent.py --text "AI is transforming industries..." --style brief
  python src/agent.py --url https://en.wikipedia.org/wiki/Artificial_intelligence --style bullet
  python src/agent.py --file docs/report.pdf --style detailed
        """,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", "-t",
        type=str,
        help="Plain text string to summarize.",
    )
    input_group.add_argument(
        "--url", "-u",
        type=str,
        help="Web URL to scrape and summarize.",
    )
    input_group.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a .txt, .pdf, or .docx file.",
    )
    parser.add_argument(
        "--style", "-s",
        type=str,
        default=DEFAULT_STYLE,
        choices=["brief", "detailed", "bullet"],
        help=f"Summary style (default: {DEFAULT_STYLE}).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point — parses args and runs the summarizer."""
    args = parse_args()
    try:
        summarize(
            text=args.text,
            url=args.url,
            file=args.file,
            style=args.style,
        )
    except (ValueError, EnvironmentError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        log.exception("Unexpected error during summarization.")
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
