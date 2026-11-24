import argparse
import contextlib
import json
import multiprocessing
import signal
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from features import graph_from_text
from maple import project
from model import PolicyNetwork
from policy import rollout


@contextlib.contextmanager
def deadline(seconds):
    def expired(*_):
        raise TimeoutError("Prediction exceeded its time budget")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def load_model(variant, checkpoint):
    torch.set_num_threads(1)
    model = PolicyNetwork(variant)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


def initialize_up(checkpoint, maple, timeout):
    global _model, _maple, _timeout
    _model, _maple, _timeout = load_model("up", checkpoint), maple, timeout


@torch.inference_mode()
def predict_up(row):
    start = time.monotonic()
    text = row["polynomials"]
    def projection(_index, prefix, state):
        nonlocal text
        selected = prefix[-1]
        remaining = [v for v in state.variables if v != selected]
        text = project(text, selected, remaining, executable=_maple,
                       timeout=_timeout - (time.monotonic() - start))
        return graph_from_text(text)
    try:
        with deadline(_timeout):
            order = rollout(_model, [graph_from_text(text)], project=projection).orders[0]
        status = "ok"
    except TimeoutError:
        order, status = [], "timeout"
    return dict(id=row["id"], order=order, status=status, seconds=time.monotonic() - start)


@torch.inference_mode()
def predict_nup(model, rows, batch_size, timeout):
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        graphs, results = [], []
        for row in batch:
            start = time.monotonic()
            result = dict(id=row["id"], order=[], status="ok")
            try:
                with deadline(timeout):
                    graphs.append(graph_from_text(row["polynomials"]))
            except TimeoutError:
                result["status"] = "timeout"
            result["seconds"] = time.monotonic() - start
            results.append(result)
        start = time.monotonic()
        orders = iter(rollout(model, graphs).orders)
        share = (time.monotonic() - start) / len(graphs) if graphs else 0
        for result in results:
            if result["status"] == "ok":
                result["order"] = next(orders)
                result["seconds"] += share
                if result["seconds"] > timeout:
                    result.update(order=[], status="timeout")
            yield result


def predict(variant, checkpoint, rows, output, *, batch_size=32, workers=4,
            maple="maple", timeout=900):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        def write(results):
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
        if variant == "nup":
            write(predict_nup(load_model(variant, checkpoint), rows, batch_size, timeout))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialize_up, initargs=(checkpoint, maple, timeout)) as pool:
                write(pool.map(predict_up, rows))


if __name__ == "__main__":
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["up", "nup"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path, help="Custom JSONL inputs instead of the test database")
    parser.add_argument("--data", type=Path, default=root / "data/test.sqlite")
    parser.add_argument("--variables", type=int, nargs="+", choices=range(3, 10), default=[3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4, help="UP projection workers")
    parser.add_argument("--maple", default="maple")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.input:
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    else:
        with sqlite3.connect(args.data.resolve().as_uri() + "?mode=ro", uri=True) as db:
            rows = [dict(id=identifier, polynomials=polynomials)
                    for variables, identifier, polynomials in db.execute(
                        "SELECT variables,id,polynomials FROM cases ORDER BY variables,id")
                    if variables in args.variables]
    predict(args.variant, args.checkpoint or root / "weights" / f"{args.variant}.pth",
            rows, args.output, batch_size=args.batch_size, workers=args.workers,
            maple=args.maple, timeout=args.timeout)
