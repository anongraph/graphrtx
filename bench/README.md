## Command Line Option

Usage: ./bench_all <graph.mtx> [OPTION

### Options

| Option | Description | Default |
|--------|------------|---------|
| `--device N` | CUDA device ID | `0` |
| `--max-seg N` | Maximum UASP segment length | `1024` |
| `--parts N` | Number of partitions (`0 = auto`) | `0` |
| `--dummies N` | Number of dummy AABBs to append | `0` |
| `--src N` | Source vertex for BFS / SSSP | `0` |
| `--pr-iters N` | Number of PageRank iterations | `20` |
| `--pr-damp F` | PageRank damping factor | `0.85` |
| `--cdlp-iters N` | Number of CDLP iterations | `20` |
| `--algo LIST` | Algorithms to run (`bfs,pr,sssp,bc,tc,wcc,cdlp,all`) | `all` |
| `--no-hybrid` | Disable hybrid algorithm variants | enabled |
| `-q, --quiet` | Minimal logging | off |
| `-h, --help` | Show help message | — |

./bench_all graph.mtx --algo bfs,pr --device 0 
