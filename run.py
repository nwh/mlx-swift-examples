from mlx_lm import load, generate

model, tokenizer = load("mlx-community/mamba-130m-hf-f32")
response = generate(model, tokenizer, prompt="hello", verbose=True)
