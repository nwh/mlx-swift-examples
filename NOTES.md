# Mamba port notes

TODO: remove this file before PR merge

From the [mlx-lm](https://github.com/ml-explore/mlx-lm) repo run:

```
mlx_lm.generate --model "mlx-community/mamba-130m-hf-f32" --prompt "the sky is"
```

To run the MLX Swift port:

```
./mlx-run llm-tool --model "mlx-community/mamba-130m-hf-f32" --prompt "the sky is"
```

Or launch `llm-tool` via Xcode for debugging.
