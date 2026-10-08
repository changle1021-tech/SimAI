"""Driver-side progress updates for completed Ray profiling tasks."""


def collect_profile_results(ray, refs, inputs, progress, describe):
    """Report completion promptly while retaining submission order in the CSV."""
    positions = {ref: i for i, ref in enumerate(refs)}
    results = [None] * len(refs)
    pending = list(refs)
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        for ref in ready:
            index = positions[ref]
            results[index] = ray.get(ref)
            if progress is not None:
                progress.set_postfix_str("done " + describe(inputs[index]), refresh=False)
                progress.update(1)
    return results
