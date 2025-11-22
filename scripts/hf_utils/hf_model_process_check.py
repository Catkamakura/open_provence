from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from transformers import AutoModel

DEFAULT_MODELS: tuple[str, ...] = (
    "hotchpotch/open-provence-reranker-v1",
    "hotchpotch/open-provence-reranker-xsmall-v1",
    "hotchpotch/open-provence-reranker-large-v1",
    "hotchpotch/open-provence-reranker-v1-gte-modernbert-base",
)

question: str = "What's your favorite Japanese food?"
context: str = """
Work deadlines piled up today, and I kept rambling about budget spreadsheets to my roommate.
Next spring I'm planning a trip to Japan so I can wander Kyoto's markets and taste every regional dish I find.
Sushi is honestly my favourite—I want to grab a counter seat and let the chef serve endless nigiri until I'm smiling through soy sauce.
Later I remembered to water the plants and pay the electricity bill before finally getting some sleep.
"""


@dataclass
class Case:
    name: str
    question: str | Sequence[str]
    # allow up to 3-level nesting: queries -> docs -> sentences
    context: str | Sequence[str] | Sequence[Sequence[str]] | Sequence[Sequence[Sequence[str]]]


@dataclass
class SampleResult:
    case: str
    sample: str
    score: float | None
    compression: float
    pruned: str | None


def build_cases() -> list[Case]:
    questions = [question, question]
    contexts = [context, context]

    context_sentences = [line for line in context.splitlines(True) if line.strip()]
    context_sentences_wrapped = [context_sentences]
    contexts_nested = [context_sentences_wrapped, context_sentences_wrapped]

    return [
        Case("q=str, c=str", question, context),
        Case("q=list[str], c=list[str]", questions, contexts),
        Case("q=str, c=list[str] (split sentences)", question, context_sentences),
        Case(
            "q=str, c=list[list[str]] (split sentences, single doc)",
            question,
            context_sentences_wrapped,
        ),
        Case(
            "q=list[str], c=list[list[str]] (split sentences per query)",
            questions,
            contexts_nested,
        ),
    ]


def _iter_samples(
    pruned_context, rerank_score, compression_rate
) -> Iterable[tuple[str, str | None, float | None, float]]:
    if not isinstance(pruned_context, list):
        yield "", pruned_context, rerank_score, compression_rate
        return

    for idx, text in enumerate(pruned_context):
        text_str = "\n".join(text) if isinstance(text, list) else text

        score = rerank_score[idx] if isinstance(rerank_score, list) else rerank_score
        compression = (
            compression_rate[idx] if isinstance(compression_rate, list) else compression_rate
        )

        if isinstance(score, list):
            score = score[0] if score else None
        if isinstance(compression, list):
            compression = compression[0] if compression else 0.0

        yield f"#{idx}", text_str, score, float(compression)


def run_cases(model, threshold: float, verbose: bool) -> list[SampleResult]:
    results: list[SampleResult] = []
    for case in build_cases():
        result = model.process(
            question=case.question,
            context=case.context,
            threshold=threshold,
            show_progress=verbose,
        )
        for sample_tag, pruned, score, compression in _iter_samples(
            result["pruned_context"],
            result["reranking_score"],
            result["compression_rate"],
        ):
            results.append(
                SampleResult(
                    case=case.name,
                    sample=sample_tag,
                    score=None if score is None else float(score),
                    compression=float(compression),
                    pruned=pruned if verbose else None,
                )
            )
    return results


def _format_table(rows: list[SampleResult]) -> str:
    headers = ["Case", "Sample", "Rerank score", "Compression"]
    data: list[list[str]] = []
    for row in rows:
        sample = row.sample or "-"
        score = "-" if row.score is None else f"{row.score:.4f}"
        compression = f"{row.compression:.2f}"
        data.append([row.case, sample, score, compression])

    col_widths = [max(len(item[i]) for item in ([headers] + data)) for i in range(len(headers))]

    def fmt_row(items: Sequence[str]) -> str:
        return " | ".join(item.ljust(col_widths[idx]) for idx, item in enumerate(items))

    divider = "-+-".join("-" * width for width in col_widths)
    lines = [fmt_row(headers), divider]
    lines.extend(fmt_row(row) for row in data)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the four HF models using the run.py sample inputs.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Hugging Face model IDs to load (default: README models).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Pruning threshold passed to model.process.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print pruned text for each sample in addition to the summary table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for model_id in args.models:
        print(f"\n=== {model_id} ===")
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        rows = run_cases(model, threshold=args.threshold, verbose=args.verbose)

        if args.verbose:
            for row in rows:
                if row.pruned is None:
                    continue
                print(f"\n-- {row.case} {row.sample or ''}".strip())
                print("Pruned context:\n" + row.pruned)
                print(f"Rerank score: {row.score}")
                print(f"Compression: {row.compression:.2f}")

        print("\n" + _format_table(rows))


if __name__ == "__main__":
    main()
