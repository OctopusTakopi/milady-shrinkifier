from __future__ import annotations

import argparse
import json

from .pipeline_common import connect_db
from .score_avatar_catalog import load_default_run_id

MODEL_LABEL_SOURCE = "model"
DEFAULT_NEGATIVE_MAX_PROBABILITY = 0.005
DEFAULT_POSITIVE_MIN_PROBABILITY = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote extreme-confidence scored avatars into lower-confidence model labels."
    )
    parser.add_argument(
        "--run-id",
        help="Model run id to read scores from. Defaults to the currently promoted run.",
    )
    parser.add_argument(
        "--max-negative-probability",
        type=float,
        default=DEFAULT_NEGATIVE_MAX_PROBABILITY,
        help="Auto-label images with score <= this threshold as not_milady.",
    )
    parser.add_argument(
        "--min-positive-probability",
        type=float,
        default=DEFAULT_POSITIVE_MIN_PROBABILITY,
        help="Optional auto-label threshold for milady. Disabled by default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of images considered, ordered by strongest confidence first.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which rows would be updated without changing the catalog.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_id = args.run_id or load_default_run_id()

    connection = connect_db()
    try:
        score_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM model_scores WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
        )
        if score_count == 0:
            raise SystemExit(f"No catalog scores found for run {run_id}. Run `uv run milady score --run-id {run_id}` first.")

        negative_candidates = connection.execute(
            """
            SELECT images.sha256, score_records.score
            FROM images
            INNER JOIN model_scores AS score_records
              ON score_records.image_sha256 = images.sha256
            WHERE score_records.run_id = ?
              AND (
                images.label IS NULL
                OR images.label_source = ?
              )
              AND score_records.score <= ?
            ORDER BY score_records.score ASC, images.sha256 ASC
            LIMIT COALESCE(?, -1)
            """,
            (run_id, MODEL_LABEL_SOURCE, args.max_negative_probability, args.limit),
        ).fetchall()

        positive_candidates = []
        if args.min_positive_probability is not None:
            positive_candidates = connection.execute(
                """
                SELECT images.sha256, score_records.score
                FROM images
                INNER JOIN model_scores AS score_records
                  ON score_records.image_sha256 = images.sha256
                WHERE score_records.run_id = ?
                  AND (
                    images.label IS NULL
                    OR images.label_source = ?
                  )
                  AND score_records.score >= ?
                ORDER BY score_records.score DESC, images.sha256 ASC
                LIMIT COALESCE(?, -1)
                """,
                (run_id, MODEL_LABEL_SOURCE, args.min_positive_probability, args.limit),
            ).fetchall()

        updates = [
            build_update_payload(run_id, str(row["sha256"]), "not_milady", float(row["score"]))
            for row in negative_candidates
        ] + [
            build_update_payload(run_id, str(row["sha256"]), "milady", float(row["score"]))
            for row in positive_candidates
        ]

        if not args.dry_run:
            connection.execute(
                """
                UPDATE images
                SET label = NULL,
                    label_source = NULL,
                    labeled_at = NULL,
                    review_notes = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE label_source = ?
                """,
                (MODEL_LABEL_SOURCE,),
            )
            for update in updates:
                connection.execute(
                    """
                    UPDATE images
                    SET label = ?,
                        label_source = ?,
                        labeled_at = CURRENT_TIMESTAMP,
                        review_notes = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE sha256 = ?
                      AND (label IS NULL OR label_source = ?)
                    """,
                    (
                        update["label"],
                        MODEL_LABEL_SOURCE,
                        update["review_note"],
                        update["sha256"],
                        MODEL_LABEL_SOURCE,
                    ),
                )
            connection.commit()

        print(
            json.dumps(
                {
                    "runId": run_id,
                    "maxNegativeProbability": args.max_negative_probability,
                    "minPositiveProbability": args.min_positive_probability,
                    "negativeModelLabels": len(negative_candidates),
                    "positiveModelLabels": len(positive_candidates),
                    "updatedLabels": len(updates),
                    "dryRun": args.dry_run,
                    "limit": args.limit,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        connection.close()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.max_negative_probability <= 1.0:
        raise SystemExit("--max-negative-probability must be between 0 and 1.")
    if args.min_positive_probability is not None and not 0.0 <= args.min_positive_probability <= 1.0:
        raise SystemExit("--min-positive-probability must be between 0 and 1.")
    if (
        args.min_positive_probability is not None
        and args.min_positive_probability <= args.max_negative_probability
    ):
        raise SystemExit("--min-positive-probability must be greater than --max-negative-probability.")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive.")


def build_update_payload(run_id: str, sha256: str, label: str, score: float) -> dict[str, str | float]:
    return {
        "sha256": sha256,
        "label": label,
        "score": score,
        "review_note": json.dumps(
            {
                "type": "model_label",
                "runId": run_id,
                "score": score,
                "predictedLabel": label,
            },
            sort_keys=True,
        ),
    }


if __name__ == "__main__":
    main()
