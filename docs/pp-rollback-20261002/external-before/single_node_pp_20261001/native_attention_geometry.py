"""vLLM head partitioning, including replicated KV heads for MQA."""
def local_attention_heads(query_heads,kv_heads,tp):
    if min(query_heads,kv_heads,tp)<1 or query_heads%tp:
        raise ValueError('Query heads must divide TP size')
    if kv_heads>=tp:
        if kv_heads%tp:raise ValueError('KV heads must divide TP size when sharded')
        local_kv=kv_heads//tp
    else:
        if tp%kv_heads:raise ValueError('TP size must divide replicated KV groups')
        local_kv=1
    return query_heads//tp,local_kv
