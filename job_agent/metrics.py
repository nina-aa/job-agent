from collections import defaultdict


def summarize(calls, graph_seconds=None):
    """Plain-text usage report from the LLM call records collected in graph state."""
    if not calls:
        return "No LLM calls made."
    by_node = defaultdict(lambda: [0, 0, 0, 0.0])
    by_model = defaultdict(int)
    failed = 0
    for c in calls:
        row = by_node[c["node"]]
        row[0] += 1
        row[1] += c["input_tokens"]
        row[2] += c["output_tokens"]
        row[3] += c["seconds"]
        by_model[c["model"]] += 1
        failed += len(c["failed_attempts"])

    lines = [f"{'node':<10}{'calls':>6}{'tokens in':>11}{'tokens out':>12}{'llm secs':>10}"]
    for node, (n, tin, tout, secs) in by_node.items():
        lines.append(f"{node:<10}{n:>6}{tin:>11}{tout:>12}{secs:>10.2f}")
    total_in = sum(c["input_tokens"] for c in calls)
    total_out = sum(c["output_tokens"] for c in calls)
    total_secs = sum(c["seconds"] for c in calls)
    lines.append(f"{'total':<10}{len(calls):>6}{total_in:>11}{total_out:>12}{total_secs:>10.2f}")
    lines.append("models used: " + ", ".join(f"{m} x{n}" for m, n in by_model.items()))
    lines.append(f"fallback hops (failed attempts before a success): {failed}")
    reasons = dict.fromkeys(a for c in calls for a in c["failed_attempts"])
    if reasons:
        lines.append("  " + "; ".join(reasons))
    if graph_seconds is not None:
        lines.append(f"graph wall time excluding human input: {graph_seconds:.2f}s")
    return "\n".join(lines)
