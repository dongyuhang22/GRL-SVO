import argparse
import json
import sqlite3
from pathlib import Path


def evaluate(database, predictions):
    indexed = {row["id"]: row for row in predictions}
    if len(indexed) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) as db:
        cases = {identifier: (variables, penalty) for identifier, variables, penalty in db.execute(
            "SELECT id,variables,penalty_cells FROM cases")}
        summaries = {}
        for variables in sorted({cases[identifier][0] for identifier in indexed}):
            cohort = {identifier: penalty for identifier, (n, penalty) in cases.items() if n == variables}
            result = dict(instances=len(cohort), predicted=0, solved=0, prediction_timeouts=0,
                          cad_failures=0, budget_exceeded=0, missing_outcomes=0)
            total_cells = total_seconds = 0.0
            for identifier, penalty in cohort.items():
                prediction = indexed.get(identifier)
                if prediction is None:
                    continue
                result["predicted"] += 1
                elapsed = prediction["seconds"]
                success = False
                if prediction["status"] == "timeout":
                    result["prediction_timeouts"] += 1
                elif prediction["status"] == "ok":
                    outcome = db.execute("SELECT cells,seconds FROM outcomes WHERE id=? AND ordering=?",
                                         (identifier, "".join(prediction["order"]))).fetchone()
                    if outcome is None:
                        result["missing_outcomes"] += 1
                        continue
                    cells, cad_seconds = outcome
                    if cells is None:
                        result["cad_failures"] += 1
                    elif elapsed + cad_seconds > 900:
                        result["budget_exceeded"] += 1
                    else:
                        success = True
                else:
                    raise ValueError(f"Unknown prediction status: {prediction['status']}")
                result["solved"] += int(success)
                total_cells += cells if success else penalty
                total_seconds += elapsed + cad_seconds if success else 900
            result["unsolved"] = result["predicted"] - result["solved"] - result["missing_outcomes"]
            complete = result["predicted"] == len(cohort) and not result["missing_outcomes"]
            result.update(complete=complete,
                          avg_cells=total_cells / len(cohort) if complete else None,
                          avg_seconds=total_seconds / len(cohort) if complete else None)
            summaries[str(variables)] = result
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/test.sqlite")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    print(json.dumps(evaluate(args.data, rows), indent=2))
