# Zindi ARC challenge
This repo summarizes the experients as part of the [Zindi ARC Africa challenge](https://zindi.africa/competitions/the-arc-challenge-africa).

![An example visualization for a task from the training dataset](imgs/train_0413.png
 "Optional Title")

## Day 1-2: EDA and problem analysis
- Exploratory data analysis (EDA)
   - Local dev setup
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
- Implemented input-copy-solver as the actual fallback baseline model
- Defined setup for LLM integration (next step!)
   - Base prompt definition 
 
Best score: 31.33% (Pub. LB, #3)

## Day 3: LLM Integration (focus)
