# Architecture Group

**Goal:** Test the limits of our Dell Pro GB10s *individually (no linking).*

- We will compare 2 different open weight model providers at various parameter sizes&nbsp;  
  (eg. 32B, 64B, 128B) and model types (MoE vs Dense)

While working on this use the ***architecture branch*** in the git repository, this is where i will put the most up to date code that may be useful for you all

```sh
git switch main
git pull --rebase
git switch architecture

# to work on your OWN code
git branch
*architecture # make sure you're on architecture branch
git switch -c <feature_branch>

# this will be a sub-branch of architecture
```

&nbsp;

### Understanding Different Models:

- **Dense:** Every parameter is used for every token. A 32B dense model reads all 32B weights each time it generates a token. This is what you think of when people say LLMs  
- [**Mixture of Experts (MoE)**](https://www.nvidia.com/en-us/glossary/mixture-of-experts/)**:** The model is split into many smaller "expert" blocks, and a router picks a few of them for each token. An MoE model has two sizes:  
  - **Total parameters** decide how much memory it needs.  
  - **Active parameters decide** how much gets read per token, and so how fast it runs.  
  - Example: `qwen3.5:122b` has 122B total but only about 10B active. It needs about 81 GB of memory but can run about as fast as a \~10B dense model.  
- [**Quantization**](https://huggingface.co/docs/optimum/en/concept_guides/quantization) (read like the first paragraph of this article): This is the format of the weights.&nbsp;  
  Think of it like using a `int vs float vs double` \- It’s the precision of the calculations

| Precision | Ollama tag suffix | GB per billion params | 32B model | 122B model |
| :---- | :---- | :---- | :---- | :---- |
| 4-bit (default) | `q4_K_M` | \~0.65 | \~21 GB | \~80 GB |
| 8-bit | `q8_0` | \~1.07 | \~34 GB | \~130 GB (won't fit) |
| 16-bit | `fp16` / `bf16` | \~2.0 | \~64 GB | \~244 GB (won't fit) |

&nbsp;

### Downloading/Running Models:

&nbsp;

```sh
# ONLY USE -q4_K_M right now
ollama pull <model_name>-q4_K_M

# see what's downloaded, and its size on disk
ollama list
```

&nbsp;

**Starting Point:** Use the built-in Ollama CLI with the `--verbose` flag to get some basic information.

```sh
# "ollama run -h" for options list

ollama run <model_name> --verbose
"
Output:
(answer...)
total duration:       2.210932368s
load duration:        1.024253ms
prompt eval count:    34 token(s)
prompt eval duration: 56.683ms
prompt eval rate:     599.83 tokens/s
eval count:           204 token(s)
eval duration:        2.115241s
eval rate:            96.44 tokens/s
"
```

&nbsp;

While a model is running you can open a second terminal:

```sh
# loaded model, memory it's using, and whether it's 100% on the GPU
ollama ps
"
NAME               ID              SIZE     PROCESSOR    CONTEXT    RUNNER      UNTIL
llama3.2:latest    a80c4f17acd5    17 GB    100% GPU     131072     llamacpp    4 minutes from now
"

# power, temperature and clock speed once a second
nvidia-smi --query-gpu=power.draw,temperature.gpu,clocks.sm --format=csv -l 1
"
49.39 W, 59, 2470 MHz
49.17 W, 59, 2470 MHz
49.42 W, 59, 2470 MHz
14.05 W, 54, 2398 MHz
11.74 W, 53, 2398 MHz
"
# unload the model when you're done (frees memory for the next one)
ollama stop <model_name>
```

&nbsp;

- Understand what each one means  
  - **What is prompt eval count, eval rate, etc.**  
  - **How should we interpret these?**

Get comfortable loading in different models and running them like this

## Assignments:

- Instead of providers it is probably better to split it my architecture, it just makes it easier to compare results  
- Brandon/Jeremy work on machine `aiml-inference.resnet.nd.edu`  
- Gabriel/Tyler work on machine `aiml_inference2.resnet.nd.edu`

| Brandon/Jeremy: Dense | Gabriel/Tyler: MoE |
| :---- | :---- |
| `llama3.1:8b` | `qwen3.5:35b-a3b` |
| `qwen3.5:9b` | `gemma4:26b` |
| `qwen3.5:27b` | `gpt-oss:20b` |
| `gemma4:31b` | `qwen3.5:122b-a10b` |
| `llama3.3:70b` | `llama4:scout` |
| `mistral-large:123b` | `gpt-oss:120b` |

&nbsp;

**Guiding Questions:**

1. What does&nbsp;
