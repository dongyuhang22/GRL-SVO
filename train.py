import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch

from features import Graph
from model import PolicyNetwork
from policy import rollout


@dataclass
class Case:
    graph: Graph
    cells: dict
    projections: dict


def decode_graph(text):
    value = json.loads(text)
    return Graph(tuple(value["variables"]),
                 torch.tensor(value["x"], dtype=torch.float32).reshape(-1, 14),
                 torch.tensor(value["edges"], dtype=torch.long).reshape(2, -1))


def load_cases(database, split, variant):
    with sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True) as db:
        cases = {identifier: Case(decode_graph(graph), {}, {})
                 for identifier, graph in db.execute(
                     "SELECT id,graph FROM cases WHERE split=? ORDER BY id", (split,))}
        for identifier, order, cells in db.execute(
                "SELECT id,ordering,cells FROM outcomes WHERE split=?", (split,)):
            cases[identifier].cells[tuple(json.loads(order))] = cells
        if variant == "up":
            for identifier, prefix, graph in db.execute(
                    "SELECT id,prefix,graph FROM projections WHERE split=?", (split,)):
                cases[identifier].projections[tuple(json.loads(prefix))] = decode_graph(graph)
    return list(cases.values())


def rollout_cases(model, cases, *, sample=False):
    def project(index, prefix, _state):
        return cases[index].projections[prefix]
    return rollout(model, [case.graph for case in cases], sample=sample,
                   project=project if model.variant == "up" else None)


def cost(cells):
    return 1.0 if cells is None else cells / 50000.0


def actor_critic_loss(log_probabilities, initial_values, costs):
    advantage = costs - initial_values
    actor = costs.new_zeros(())
    critic = costs.new_zeros(())
    for i in range(len(costs)):
        actor = actor + advantage[i].detach() * log_probabilities[i] / len(costs)
        critic = critic + advantage[i].square() / len(costs)
    return actor + critic


@torch.inference_mode()
def validate(model, cases, batch_size):
    model.eval()
    cells = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        result = rollout_cases(model, batch)
        cells.extend(case.cells[tuple(order)] for case, order in zip(batch, result.orders))
    return sum(value is None for value in cells), sum(map(cost, cells)) / len(cells)


def train(database, variant, output, *, epochs=100, batch_size=32, lr=2e-5, seed=2022):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train_cases = load_cases(database, "train", variant)
    valid_cases = load_cases(database, "valid", variant)
    model = PolicyNetwork(variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_score, best_epoch = None, None
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(train_cases)).tolist()
        loss_sum = 0.0
        for start in range(0, len(indices), batch_size):
            batch = [train_cases[i] for i in indices[start:start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            result = rollout_cases(model, batch, sample=True)
            costs = result.initial_values.new_tensor([
                cost(case.cells[tuple(order)]) for case, order in zip(batch, result.orders)])
            loss = actor_critic_loss(result.log_probabilities, result.initial_values, costs)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        record = dict(epoch=epoch, loss=loss_sum / len(train_cases))
        score = validate(model, valid_cases, batch_size)
        if best_score is None or score < best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), output / "best.pth")
        record.update(validation_failures=score[0], validation_mean_cells=score[1] * 50000,
                      best_epoch=best_epoch)
        print(json.dumps(record), flush=True)
    torch.save(model.state_dict(), output / "last.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["up", "nup"], required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/training.sqlite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()
    train(args.data, args.variant, args.output, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, seed=args.seed)
