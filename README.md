# Zindi ARC challenge: A 5-day Sprint
This repo summarizes the experients as part of the [Zindi ARC Africa challenge](https://zindi.africa/competitions/the-arc-challenge-africa).

This competition can serve as a starting point for anyone willing to explore the [ARC-AGI](https://arcprize.org/) challenges. 

Goal: to be ranked among the top 10 participants at the end of the competition.

![An example visualization for a task from the training dataset](imgs/train_0413.png
 "Optional Title")

 ## Usage
 Reproducing the competiton score can be done by using the fallback solver through the [zindi-arc-solver.ipynb](/zindi-arc-solver.ipynb) notebook from top to bottom. The below description is also sumarized in the `solution.pdf` document. The solution was only tested on Unix-based host machines.

 Note: If running the submission notebook on Google colab, make sure to use the following steps:
 - upload the `zindi-arc-solver-colab.ipynb` to colab; no GPU is required
 - After running the cell containing the command `!mkdir submissions src data`, upload the data and src files to their respected folder
    - data/
        - train.json
        - test.json
        - SampleSubmission.csv
    - src/
        - `__init__.py`
        - model.py
        - utils.py
        - config.py
        - base_prompt.txt
  - update the `config.py` file by replacing the root folder (if necessary)
 ```python 
 root="/content/" # or keep it as "./"
 ```
 - Then run the subsequent cells to generate the submission file


## Day 1-2: EDA and problem analysis
The EDA stage can be replecated by running the [data-exploration.ipynb](/data-exploration.ipynb) notebook.

- Exploratory data analysis (EDA)
   - Local dev setup (MBP M4 - 24GB, NVIDIA RTX 6000 - 48GB)
   - Data visualization toolkit built around the code base from [Oleg X@Kaggle](https://www.kaggle.com/code/allegich/arc-agi-2025-starter-notebook-eda/notebook)
   - Added and refined utility functions for: 
      - Grid analysis and manipulation
      - Random data sampling
      - Submission formatting
      - Model definition and puzzle solving
      - Local evaluation (exact match, pixel value accuracy).

### Insights
A. Data structure
- The examples are stored as keys in the raw `JSON` file
- Each example has a `train` and a `test` key
- Puzzle instances which we refer to as `problems or problem sets` are stored a list in the `train` attribute. 
    - Each problem is comprised of `input-output` pairs (at least 2; and no more than 8) to extract the problem logic from.
    - Both the inputs and outputs are grids represented in matrix form as lists of lists. These lists are filled with numbers, each representing a particular color of the grid cells from a set of 10 colors.
        - Allowed colors are: # 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
        - The smallest possible grid size is `1x1` and the largest is `30x30`

    - A test case is provided to try out the inferred logic on (or `solve`) the problem.
    - summary of problem structure (train examples)
  
```
train_xxx:
        # these are to be used to learn the problem's logic 
            - [
                {
                    "input": [[...], [...]...[...]], # this is a problem
                    "output": [[...], [...]...[...]] # this is the corresponding solution
                },
                {
                    "input": [[...], [...]...[...]], 
                    "output": [[...], [...]...[...]]
                },
                ...
                {
                    "input": [[...], [...]...[...]], 
                    "output": [[...], [...]...[...]]
                }
            ]
        test_xxx:
        # Use the infer logic to solve this problem
            - [
                {
                    "input": [[...], [...]...[...]], # this is the problem to be solved
                    "output": [[...], [...]...[...]] # this is the corresponding solution (only provided in the training subset)
                }
            ]            
```

B. Baseline identification and initial experiments
- The solution to a puzzle may have a grid of a size similar, bigger, or smaller than that ot the test input.
- The most common background in the training and test sets seems to be #7 (orange). This can motivate the use of that value as the default fill value instead of #0 (black).
    - With Os: `0.310%` (Pub. LB)
    - With 7s: about `11.72 (Pub. LB)` => using `zeros as background pixels hurts accuracy on the competition test set` for this challenge, especially given the evaluation strategy. Thus, a basic and lazy intuition while building the solver is to fall back to orange background pixels if the underlying rule(s) cannot be inferred from the training examples.
- This hackathon is `much much easier` that the original ARC-AGI challenge (perhaps intentionally) as we also know a few things about the test data, which makes is pretty much #hackable in the sense that the expected predictions are bounded:
    -  We know the expected number of rows, so the major task on the grid aspect is to determine the expected number of columns. In fact, this might be a design choice for the use of a specific evaluation metric. `Ignoring this might result in several mismatch issues!!`
    - This may help participants in setting a more realistic baseline to check whether or not the implemented models are doing better than a lazy solver (e.g. background predictor which scores about `4.65%` on the Pub. LB).
    - We also know the expected number of columns, but I intentionally ignored this detail while designing the solution

Best score: 11.72% (Pub. LB)

## Day 2: Baseline Experiments (focus)
- Implemented heuristics for grid/column inference: `infer_out_shape`
   - Identical shapes for inp-out pairs
   - All outputs with the same shape
   - Scaling by a certain factor (*)
   - Exact match with a training example (*)
   - Using test input shape (*)
   - A few fixes are still remaining (fallback options are a bit weak)
   - We intentionally refrain from also fixing the expected number of columns as per the competition's submission file;
- Implemented input-copy-solver as the actual fallback baseline model
- Defined setup for LLM integration (next step!)
   - Base prompt definition 
 
Best score: 31.33% (Pub. LB)

## Day 3-4: LLM Integration (focus)
Research questions: 
- Does leveraging an LLM improve the model's performance?
- Which base LLMs provide better `performance-cost` compromise?

### Base models:

`Mistral-Nemo-Instruct-2407-4bit` & `Mistral-Nemo-Instruct-2407-4bit`
- Pretty solid performance and easily integrated
- No reasoning but fairly good analysis and easy prompt-follower
- 4-bit version struggles with large input shapes (greater than 20x20) frem experiments but relatively fast
- 8-bit version is a bit slow 

`Phi-4-mini-reasoning-8bit`
- Thinking model; In-depth analysis of the problem
- Too verbose (too much thinking), even when explicitly asked not to.

`Qwen/Qwen2.5-14B-Instruct`
- ~15GB on Mac (8-bit); ~29GB (~22GB using 8-bit and ~16GB using 4-bit) on RTX 6000
    - Quantizing to 8-bit allows for keeping the model within the 24GB limit, making it possible to run it e.g. on an RTX 4090 
    ```python
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, # requires less than 16GB of VRAM
        load_in_8bit=False, # if true, requires less than 24GB of VRAM
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True
    )

    llm = AutoModelForCausalLM.from_pretrained(
        base_llm_name, 
        torch_dtype=Config.dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code   = Config.trust_remote_code,
        quantization_config=quantization_config
    )
    ```
- Very fast (CUDA) and consistent with the solving process
- Cannot solve larger inputs on M4 by default (crashes for grid size larger than 15x15)
- Running on the full test set takes ~2hrs in 4-bit or 8-bit precision, each output being generated in a maximum of `3-5min` in a single pass.
- No extra prompting or sophisticated tuning
- `Pub. LB: 28.846%` with baseline setup (pre-sealing)
    - While preparing my code for review, I noticed a mistake in the grid extraction logic `src/model.py - extract_out_grid` that seems to have affected the model's prediction workflow by replacing most of the predicted values with the `DEFAULT_BG_VALUE`.
    - Fixing this bug resulted in a `Pub. LB score of 32.382% (Priv. LB: 31.958%)`, an over`+3`improvement on the originally reported score and a `+1` improvement on the previous best score (cp-solver)
    - You can run this experiment with the [cuda-llm-solver.ipynb](/cuda-llm-solver.ipynb) notebook

## TODO LIST
- Try Qwen2.5 with better prompting and an addtion of tool use (specific functions for grid inference)
    - Also after bug fixes, optimize pipeline for faster inference
- Clean up code base to remove competition-specific conditioning after reviews
    - to facilitate exploration of the official challenge
- Explore multimodal LLM and computer vision approaches 
    - Intuition: too hard to textually describe the intermediate steps, as they usually appear obvious from a visual perspective.

## Acknowledgement
This project was developed in close collaboration with GPT-4o as a coding assistant. 
Its contributions included brainstorming experimental directions, assisting with iterative code completion and refactoring, and supporting the development of grid evaluation routines, prompt engineering, and strategy testing. 
The interactive workflow allowed for rapid prototyping and refinement, significantly accelerating the pace of experimentation, given the short-term setting.

I would like to express my gratitude to the organizers for setting up this challenge, which not only provided a stimulating and well-structured benchmark, but also rekindled my engagement with the Zindi platform after a long break. 
The competition offered a refreshing opportunity to explore abstract reasoning tasks in a focused, time-constrained setting.
