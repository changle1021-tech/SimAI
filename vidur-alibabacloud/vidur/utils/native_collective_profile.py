"""Transport-qualified native collective GPU primitives."""
import bisect,math,statistics

class NativeCollectiveProfile:
    def __init__(self, rows, embedding_dim, device=None):
        self.embedding_dim=embedding_dim;grouped={}
        for row in rows:
            if device and device.lower() not in row.get('device_name','').lower():continue
            if row['dtype']!='float16':raise ValueError('Native collective dtype is unsupported')
            value=float(row['time_stats.all_reduce.mean'])
            if not math.isfinite(value) or value<=0:raise ValueError('Invalid native collective GPU span')
            key=(row['execution_mode'],int(row['size']))
            grouped.setdefault(key,[]).append(value)
        self.values={k:statistics.mean(v) for k,v in grouped.items()}
        if {k[0] for k in self.values}!={'eager','cuda_graph'}:
            raise ValueError('Native collective profiles need eager and CUDA graph execution modes')
    def predict(self,tokens,use_eager):
        mode='eager' if use_eager else 'cuda_graph';size=tokens*self.embedding_dim*2
        grid=sorted(k[1] for k in self.values if k[0]==mode)
        if size<grid[0] or size>grid[-1]:raise ValueError('Native collective profile lacks message-size coverage')
        i=bisect.bisect_left(grid,size)
        if grid[i]==size:return self.values[(mode,size)]
        lo,hi=grid[i-1:i+1];fraction=(size-lo)/(hi-lo)
        return self.values[(mode,lo)]*(1-fraction)+self.values[(mode,hi)]*fraction
