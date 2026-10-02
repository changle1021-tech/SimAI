"""CPU producer / CUDA stream consumer recurrence for eager model forwards."""
import math


def replay_program(program, gpu_duration):
    """CUDA commands become ready as their CPU producer finishes submitting.

    GPU primitives come from independent operator profiles. CPU submission can
    overlap previous GPU work; a starved stream waits for the next submission.
    call-end readiness conservatively represents a module with multiple kernel
    submissions. No measured forward span or request E2E is used as a target.
    """
    host, stream = 0.0, 0.0
    for frame in program['frames']:
        gap, call = frame['gap_ms'], frame['call_ms']
        duration = gpu_duration(frame['operation'], frame['layer'])
        if any(not math.isfinite(x) or x < 0 for x in (gap, call, duration)):
            raise ValueError('Invalid CPU/GPU primitive duration')
        host += gap + call
        stream = max(stream, host) + duration
    tail = program['tail_ms']
    if not math.isfinite(tail) or tail < 0:
        raise ValueError('Invalid eager CPU tail')
    return max(stream, host + tail)


def interpolate_program(points, batch_size, prompt_tokens):
    """Bounded interpolation of CPU submission programs at a fixed batch size."""
    available=sorted(p for b,p in points if b==batch_size)
    if not available or not available[0]<=prompt_tokens<=available[-1]:
        raise ValueError('Eager CPU dispatch profile does not cover this batch/prompt')
    if (batch_size,prompt_tokens) in points:
        return points[(batch_size,prompt_tokens)]
    lo=max(p for p in available if p<prompt_tokens)
    hi=min(p for p in available if p>prompt_tokens)
    left,right=points[(batch_size,lo)],points[(batch_size,hi)]
    if [(f['operation'],f['layer']) for f in left['frames']]!=[(f['operation'],f['layer']) for f in right['frames']]:
        raise ValueError('CPU dispatch programs have different operation layouts')
    weight=(prompt_tokens-lo)/(hi-lo)
    mix=lambda a,b:a+(b-a)*weight
    return dict(frames=[dict(operation=a['operation'],layer=a['layer'],
                      gap_ms=mix(a['gap_ms'],b['gap_ms']),
                      call_ms=mix(a['call_ms'],b['call_ms']))
                      for a,b in zip(left['frames'],right['frames'])],
                tail_ms=mix(left['tail_ms'],right['tail_ms']))
