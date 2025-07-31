# Zindi ARC challenge: A 5-day Sprint
This repo summarizes the experients as part of the [Zindi ARC Africa challenge](https://zindi.africa/competitions/the-arc-challenge-africa).

The competition can serve as a cool starting point for anyone willing to explore the [ARC-AGI](https://arcprize.org/) challenges. 

PS: For the purpose of the Zindi competition, I'm interested in how far can low-cost (often extremely custom) approaches can lead in terms of accuracy. My initial hypothesis, based on the design of the competition is that `a better understanding of the dataset gives enough insights for building a basic solver that can score up to 50% on the public leaderboread`...and hope to be able to demonstrate it, through an iterative process. As such, I will not be exploring any approaches other than few-shot prompting. 

Goal: to be ranked among the top 10 participants at the end of the competition.

![An example visualization for a task from the training dataset](imgs/train_0413.png
 "Optional Title")

## Day 1-2: EDA and problem analysis
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
    - With zeros: `0.310%` (Pub. LB)
    - With sevens: about `11.72 (Pub. LB)` => using `zeros as background pixels is counterproductive` for this challenge, especially given the evaluation strategy. Thus, a basic and lazy intuition while building the solver is to fall back to orange background pixels if the underlying rule(s) cannot be inferred from the training examples.
- This hackathon is `much much easier`(perhaps intentionally) as we also know a few things about the test data, which makes is pretty much #hackable in the sense that the expected predictions are bounded:
    -  We know the expected number of rows, so the major task on the grid aspect is to determine the expected number of columns. In fact, this might be a design choice for the use of a specific evaluation metric. `Ignoring this might result in several mismatch issues!!` Plus this helps the participants in setting a more realistic baseline to check whether or not the implemented models are doing better than a lazy solver (background predictor which scores about `4.65%` on the Pub. LB).

Best score: 11.72% (Pub. LB, #10)

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
 
Best score: 31.33% (Pub. LB, #3)

## Day 3-4: LLM Integration (focus)
Research questions: 
- Does leveraging an LLM improve the score model's performance?
- Which base LLMs provide better `performance-cost` compromise?

### Base models:

Mistral-Nemo-Instruct-2407-4bit & Mistral-Nemo-Instruct-2407-4bit
- Pretty solid performance and easily integrated
- No reasoning but fairly good analysis and easy prmpt-follower
- 4-bit version struggles with large input shapes (greater than 20x20) frem experiments but relatively fast
- 8-bit version is a bit slow 

Phi-4-mini-reasoning-8bit
- Thinking model; In-depth analysis of the problem
- Too much thinking, even when explicitly asked not to.

Qwen/Qwen2.5-14B-Instruct
- ~15GB on MBP (8-bit); ~29GB on RTX 6000
- Very fast and consistent with the solving the puzzles
- Pub. LB: 28.846% with baseline setup


## TODO
- Try Qwen2.5 with better prompting and an addtion of tool use (specific functions for grid inference)
- clean up code base to remove competition-specific conditioning
- Explore multimodal computer vision approaches 

## Acknowledgement
This codebase was developed with the collaboration of GPT-4o:
- Brainstorming
- Code completion
- Code Refactoring