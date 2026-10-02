"""Build command lines for the isolated HTTP profiling server and client."""

import sys


PROFILE_MODEL_NAME = 'vidur-profile-model'


def server_arguments(args, server_script, output_dir):
    """Return the vLLM server command using the requested profile arguments."""
    command = [
        sys.executable, str(server_script),
        '--diagnostic-output', str(output_dir),
        '--model', str(args.model),
    ]
    if args.tokenizer is not None:
        command.extend(['--tokenizer', str(args.tokenizer)])
    if args.revision is not None:
        command.extend(['--revision', str(args.revision)])
    if args.tokenizer_revision is not None:
        command.extend(['--tokenizer-revision', str(args.tokenizer_revision)])

    command.extend([
        '--served-model-name', PROFILE_MODEL_NAME,
        '--dtype', str(args.dtype),
        '--load-format', str(args.load_format),
        '--seed', str(args.seed),
        '--max-model-len', str(args.max_model_len),
        '--tensor-parallel-size', str(args.tensor_parallel_size),
        '--pipeline-parallel-size', str(args.pipeline_parallel_size),
        '--max-num-seqs', str(args.max_num_seqs),
        '--max-num-batched-tokens', str(args.max_num_batched_tokens),
        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
        '--distributed-executor-backend', 'ray',
        '--disable-custom-all-reduce',
        '--disable-log-stats',
        '--host', '127.0.0.1',
        '--port', '8089',
        '--block-size', '16',
        '--max-seq-len-to-capture', str(args.max_model_len),
    ])
    if args.enforce_eager:
        command.append('--enforce-eager')
    if args.trust_remote_code:
        command.append('--trust-remote-code')
    return command


def client_arguments(args, client_script, server_output, output_file):
    """Return the HTTP profile client command for the requested cases."""
    command = [
        sys.executable, str(client_script),
        '--server-output', str(server_output),
        '--model-name', PROFILE_MODEL_NAME,
        '--model-name-for-vidur', str(args.model_name_for_vidur or args.model),
        '--tp', str(args.tensor_parallel_size),
        '--pp', str(args.pipeline_parallel_size),
        '--batch-sizes', *(str(value) for value in args.batch_sizes),
        '--repetitions', str(args.repetitions),
        '--network-transport', str(args.network_transport or 'local'),
        '--output', str(output_file),
        '--prompt-tokens', str(args.prompt_tokens),
        '--decode-tokens', str(args.decode_tokens),
        '--diagnostic-request-interval', str(getattr(args,'diagnostic_request_interval',0)),
    ]

    if args.enforce_eager:
        command.append("--enforce-eager")
    return command
