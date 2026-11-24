from dataclasses import dataclass

import numpy as np
import torch

from features import Graph
from model import PolicyNetwork, greedy_action


@dataclass
class Trajectory:
    orders: list[list[str]]
    log_probabilities: torch.Tensor
    initial_values: torch.Tensor


def forward_graphs(model: PolicyNetwork, graphs: list[Graph]):
    sizes = [len(g.variables) for g in graphs]
    device = graphs[0].x.device
    offsets, offset = [], 0
    for size in sizes:
        offsets.append(offset)
        offset += size
    x = torch.cat([g.x for g in graphs])
    edges = torch.cat([g.edge_index + shift for g, shift in zip(graphs, offsets)], dim=1)
    batch = torch.repeat_interleave(torch.arange(len(graphs), device=device),
                                    torch.tensor(sizes, device=device))
    logits, values = model(x, edges, batch)
    return list(logits.flatten().split(sizes)), values.flatten()


def choose_actions(scores: list[torch.Tensor], *, sample: bool):
    probabilities = [part.reshape(-1, 1).softmax(0) for part in scores]
    if sample:
        indices = [int(np.random.choice(len(p), p=p.detach().flatten().cpu().numpy()))
                   for p in probabilities]
    else:
        indices = [greedy_action(part) for part in scores]
    return indices, torch.stack([p[index, 0] for p, index in zip(probabilities, indices)])


def delete_variable(model: PolicyNetwork, graph: Graph, action: int) -> Graph:
    keep = torch.arange(len(graph.variables), device=graph.x.device) != action
    mapping = keep.long().cumsum(0) - 1
    neighbours = graph.edge_index[1, graph.edge_index[0] == action]
    edges = graph.edge_index[:, keep[graph.edge_index[0]] & keep[graph.edge_index[1]]]
    edges = mapping[edges]
    neighbours = set(neighbours.tolist())
    x = torch.stack([
        model.neighbor(torch.cat((graph.x[i], graph.x[action]))) if i in neighbours else graph.x[i]
        for i in range(len(graph.variables)) if i != action
    ])
    return Graph(tuple(v for i, v in enumerate(graph.variables) if i != action), x, edges)


def rollout(model: PolicyNetwork, graphs: list[Graph], *, sample=False, project=None) -> Trajectory:
    if model.variant == "up" and project is None:
        raise ValueError("UP requires a projection backend")
    if not graphs:
        empty = next(model.parameters()).new_zeros(0)
        return Trajectory([], empty, empty)
    states = [Graph(g.variables, model.origin(g.x), g.edge_index) if model.variant == "nup" else g
              for g in graphs]
    orders = [list(g.variables) if len(g.variables) <= 1 else [] for g in graphs]
    probabilities = graphs[0].x.new_ones(len(graphs))
    initial_values = graphs[0].x.new_zeros(len(graphs))
    first_step = True
    while True:
        active = [i for i, state in enumerate(states) if state is not None and len(state.variables) > 1]
        if not active:
            break
        scores, values = forward_graphs(model, [states[i] for i in active])
        actions, step_probabilities = choose_actions(scores, sample=sample)
        indices = torch.tensor(active, device=step_probabilities.device)
        probabilities = probabilities.index_copy(0, indices, probabilities[indices] * step_probabilities)
        if first_step:
            initial_values = initial_values.index_copy(0, indices, values)
            first_step = False
        for i, action in zip(active, actions):
            state = states[i]
            selected = state.variables[action]
            orders[i].append(selected)
            remaining = [v for v in state.variables if v != selected]
            if len(remaining) == 1:
                orders[i].extend(remaining)
                states[i] = None
            elif model.variant == "nup":
                states[i] = delete_variable(model, state, action)
            else:
                graph = project(i, tuple(orders[i]), state)
                orders[i].extend(sorted(set(remaining) - set(graph.variables)))
                if len(graph.variables) <= 1:
                    orders[i].extend(graph.variables)
                    states[i] = None
                else:
                    states[i] = graph
    return Trajectory(orders, probabilities.log(), initial_values)
