"""
# Author Information
======================
Author: Cedric Manouan
Last Update: 2 Aug, 2025
"""

from src.__init__ import logger

from collections import defaultdict, Counter

from src.config import Config
import csv

import json

import matplotlib.pyplot as plt


import os.path as osp

import pandas as pd

import random 
import re

from typing import List

def load_data(data_path:str):
    """Load training and test data"""
    # Load training data (includes solutions for learning)
    with open(osp.join(data_path, 'train.json'), 'r') as f:
        train_data = json.load(f)

    # Load test data (no solutions)
    with open(osp.join(data_path, 'test.json'), 'r') as f:
        test_data = json.load(f)

    return train_data, test_data

def sample_arc_task(data_dict:dict, split:str="train"):
    """
    Samples a random ARC task from a dictionary of ARC problems.

    Args:
        data_dict (dict): Dictionary with keys like 'train_xxx', 'test_xxx', etc.
                         Each value is a list with one item: a dict with 'train' and 'test' keys.

        split (str): "train" or "test"
    Returns:
        tuple: (task_id, task_dict), where task_dict contains 'train' and 'test' keys.
    """
    task_id = random.choice(list(data_dict.keys()))
    
    if split is not None:
        task_data = data_dict[task_id][split]
    else:
        task_data = data_dict[task_id]
        
    return task_id, task_data

def plot_grid(
        ax, 
        task:str, 
        split:str, 
        instance_type:str,
        idx, 
        input_matrix=None,
        w=0.8
    ):
    """
        idx: (int) index of (train) sample to plot
        task: (str) task id
        split: (str) train or test
        instance_type: (str) input or output 
    """
    
    fs=12

    if input_matrix==None:
        input_matrix = task[split][idx][instance_type]
        
    ax.imshow(input_matrix, cmap=Config.CMAP, norm=Config.NORM)
    
    # ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 1.0)
    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    
    '''Grid:'''
    ax.grid(visible= True, which = 'both', color = '#666666', linewidth = w)
    
    ax.tick_params(axis='both', color='none', length=0)
   
    '''sub title:'''
    ax.set_title(split + ' ' + instance_type, fontsize=fs, color = '#dddddd')

def plot_random(
    data,
    size=1.8,
    w1=0.9,
    titleSize=16    
):
    
    num_train = 2 # plotting max 2 training examples
    num_test  = 1
    wn=num_train+num_test

    task_id, task = sample_arc_task(data, split=None)
    
    fig, axs  = plt.subplots(2, wn, figsize=(size*wn,2*size))
    plt.suptitle(f'Task {task_id}', fontsize=titleSize, fontweight='bold', y=1, color = '#eeeeee')

    '''train:'''
    for j in range(num_train):
        plot_grid(ax=axs[0, j], idx=j, task=task, split='train', instance_type='input',  w=w1)
        plot_grid(ax=axs[1, j], idx=j, task=task, split='train', instance_type='output', w=w1)

    '''test:'''
    k = 0
    plot_grid(ax=axs[0, j+k+1], idx=k, task=task, split='test', instance_type='input', w=w1)
    if "output" in task["test"][0].keys():
        plot_grid(ax=axs[1, j+k+1], idx=k, task=task, split='test', instance_type='output', w=w1)

    axs[1, j+1].set_xticklabels([])
    axs[1, j+1].set_yticklabels([])
    axs[1, j+1] = plt.figure(1).add_subplot(111)
    axs[1, j+1].set_xlim([0, wn])

    '''Separators:'''
    colorSeparator = 'white'
    for m in range(1, wn):
        axs[1, j+1].plot([m,m],[0,1],'--', linewidth=1, color = colorSeparator)
    axs[1, j+1].plot([num_train,num_train],[0,1],'-', linewidth=3, color = colorSeparator)

    axs[1, j+1].axis("off")

    '''Frame and background:'''
    fig.patch.set_linewidth(5) #widthframe
    fig.patch.set_edgecolor('black') #colorframe
    fig.patch.set_facecolor('#444444') #background

    plt.tight_layout()
    plt.show() 

def visualize_task(
        task_data, 
        task_id, 
        task_solutions=None, 
        size=2.5,
        w1=0.9
    ):

    """
        Adapted from https://www.kaggle.com/code/allegich/arc-agi-2025-starter-notebook-eda/notebook
        courtesy of Oleg X@Kaggle
    """
    
    titleSize=16    
    num_train = len(task_data['train'])
    num_test  = len(task_data['test'])
    
    wn=num_train+num_test
    fig, axs  = plt.subplots(2, wn, figsize=(size*wn,2*size))
    plt.suptitle(f'Task #{task_id}', fontsize=titleSize, fontweight='bold', y=1, color = '#eeeeee')
   
    '''train:'''
    for j in range(num_train):     
        plot_grid(ax=axs[0, j], idx=j,task=task_data, split='train', instance_type='input',  w=w1)
        plot_grid(ax=axs[1, j], idx=j,task=task_data, split='train', instance_type='output', w=w1)
    
    '''test:'''
    for k in range(num_test):
        plot_grid(ax=axs[0, j+k+1], idx=k, task=task_data, split='test', instance_type='input', w=w1)
        
        if task_solutions is not None:
            task_data['test'][k]['output'] = task_solutions[task_id]
        
        if "output" in task_data["test"][0].keys():
            plot_grid(ax=axs[1, j+k+1], idx=k, task=task_data, split='test', instance_type='output', w=w1)
        
    axs[1, j+1].set_xticklabels([])
    axs[1, j+1].set_yticklabels([])
    axs[1, j+1] = plt.figure(1).add_subplot(111)
    axs[1, j+1].set_xlim([0, wn])
    
    '''Separators:'''
    colorSeparator = 'white'
    for m in range(1, wn):
        axs[1, j+1].plot([m,m],[0,1],'--', linewidth=1, color = colorSeparator)
    axs[1, j+1].plot([num_train,num_train],[0,1],'-', linewidth=3, color = colorSeparator)

    axs[1, j+1].axis("off")

    '''Frame and background:'''
    fig.patch.set_linewidth(5) #widthframe
    fig.patch.set_edgecolor('black') #colorframe
    fig.patch.set_facecolor('#444444') #background
   
    plt.tight_layout()
    plt.show() 

def get_grid_shape(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    return rows, cols

def grid_to_text(grid: List[List[int]]) -> str:
    """Convert a numeric grid to text representation"""
    lines = []
    for row in grid:
        row_str = ' '.join(str(cell) for cell in row)
        lines.append(row_str)
    return '\n'.join(lines)

def text_to_grid(text: str) -> List[List[int]]:
    """Convert text back to numeric grid"""
    try:
        lines = text.strip().split('\n')
        grid = []

        for line in lines:
            # Clean the line and extract numbers
            cleaned = line.strip()
            if cleaned and any(c.isdigit() for c in cleaned):
                # Extract all numbers from the line
                numbers = []
                current_num = ''

                for char in cleaned + ' ':
                    if char.isdigit():
                        current_num += char
                    elif current_num:
                        numbers.append(int(current_num))
                        current_num = ''

                if numbers:
                    grid.append(numbers)

        # fix grid in case not rectangular
        grid = fill_irregular_grid(grid)
        
        return grid if grid else [[0]]
    except Exception as e:
        logger.info(f"Error parsing grid: {e}")
        return [[0]]
    
def get_row_sizes(df_:pd.DataFrame):

    df = df_.copy()
    
    df["ID"] = df["ID"].astype(str)
    
    # Split only on the last underscore
    split_df = df["ID"].str.rsplit("_", n=1, expand=True)
    
    # Rename for clarity and drop malformed rows
    split_df.columns = ["ID_", "n_rows"]
    split_df = split_df.dropna()
    split_df["n_rows"] = pd.to_numeric(split_df["n_rows"], errors="coerce")
    split_df = split_df.dropna(subset=["n_rows"])
    
    # Add back to original DataFrame if needed
    df = df.join(split_df)
    
    # Group by prefix and get max suffix
    max_rows = df.groupby("ID_")["n_rows"].max().reset_index()

    return max_rows

def reconstruct_grids(df):
    """
    Reconstructs 2D grids from a dataframe with columns 'ID' and 'row'.
    
    Returns:
        Dict[str, List[List[int]]]: {task_id: grid}
    """
    grouped = defaultdict(list)

    for _, row in df.iterrows():
        task_prefix = "_".join(row["ID"].split("_")[:2])
        row_values = [int(c) for c in row["row"].strip()]
        grouped[task_prefix].append(row_values)

    return dict(grouped)

def infer_out_shape(
        train_pairs, 
        test_input, 
        expected_rows=None,
        verbose:bool=True
    ):
    """
    Infers the expected output shape for a test input, based on training input-output pairs.

    Args:
        train_pairs (List[Dict]): Each dict has 'input' and 'output' grids.
        test_input (List[List[int]]): The test input grid.
        expected_rows (int, optional): If known, override row count in output shape.

    Returns:
        Tuple[int, int]: (rows, cols) of expected output grid.
    """
    
    input_shapes            = [get_grid_shape(p["input"]) for p in train_pairs]
    output_shapes           = [get_grid_shape(p["output"]) for p in train_pairs]
    test_shape              = get_grid_shape(test_input)
        
    if all(inp == out for inp, out in zip(input_shapes, output_shapes)):
        if verbose:
            logger.info("Input and output shapes are identical in all pairs")
        return (expected_rows if expected_rows else test_shape[0], test_shape[1])

    if len(set(output_shapes)) == 1:
        if verbose:
            logger.info("All outputs have same shape")
        rows, cols = output_shapes[0]
        if expected_rows:
            rows = expected_rows
        return (rows, cols)

    for (train_in_shape, train_out_shape) in zip(input_shapes, output_shapes):
        if test_shape == train_in_shape:
            if verbose:
                logger.info("Test input shape matches a training input shape exactly")  
            rows = expected_rows if expected_rows else train_out_shape[0]
            return (rows, train_out_shape[1])   
        
    if verbose:
        logger.info("Looking for constant scaling between input and output")
    scales = []
    for (in_r, in_c), (out_r, out_c) in zip(input_shapes, output_shapes):
        if in_r != 0 and in_c != 0:
            scales.append((out_r / in_r, out_c / in_c))
    
    if scales and all(s == scales[0] for s in scales):
        scale_r, scale_c = scales[0]
        return (
            expected_rows if expected_rows else round(test_shape[0] * scale_r),
            round(test_shape[1] * scale_c)
        )
    else:
        if verbose:
            logger.info("Inconsistency detected...exploring other options.") 
    
    if verbose:
        logger.info("Use test input shape with expected_rows override")    
    rows = expected_rows if expected_rows else test_shape[0]
    
    return (rows, test_shape[1])

def fill_irregular_grid(
    grid, 
    use_mode_if_none=True
):
    """
    Makes a non-rectangular grid rectangular by padding short rows.

    Args:
        grid (List[List[int]]): A 2D list with possibly unequal row lengths.
        use_mode_if_none (bool): If True and fill_value is None, use the most common value.

    Returns:
        List[List[int]]: A rectangular grid (all rows same length).
    """
    if not grid:
        return []

    row_lengths = [len(row) for row in grid]
    if len(set(row_lengths)) == 1:
        return grid  # already regular

    max_cols = max(row_lengths)

    if use_mode_if_none:
        flat = [v for row in grid for v in row]
        fill_value = Counter(flat).most_common(1)[0][0] if flat else Config.DEFAULT_BG_VALUE

    filled = [
        row + [fill_value] * (max_cols - len(row)) if len(row) < max_cols else row
        for row in grid
    ]

    return filled

def sort_K(id_str):
    match = re.match(r"(.*)_(\d+)$", id_str)
    if match:
        task_id, row_num = match.groups()
        return (task_id, int(row_num))
    return (id_str, 0)  # fallback

def generate_submission(
    test_data,
    predictions, 
    expected_rows_list: pd.DataFrame, 
    output_path="submission.csv",
    fill_value=Config.DEFAULT_BG_VALUE
):
    """
    Generates a CSV submission file by solving all test tasks.
    Pads missing rows and sorts by ID before saving.
    """
    assert len(predictions) == len(expected_rows_list), "Mismatch between tasks and expected row counts"

    submission_rows = []
    test_keys = expected_rows_list.ID_.tolist()

    for idx, task_id in enumerate(test_keys):
        expected_rows = expected_rows_list.n_rows.iloc[idx]
        pred = predictions[idx]

        n_cols = len(pred[0]) if pred else expected_rows  # avoid crash on empty predictions

        # Pad missing rows if needed
        while len(pred) < expected_rows:
            pred.append([fill_value] * n_cols)

        # Add to submission
        for row_idx, row in enumerate(pred):
            row_str = "".join(str(cell) for cell in row)
            submission_rows.append((f"{task_id}_{row_idx+1}", row_str))

    # Sort by ID
    submission_rows.sort(key=lambda x: sort_K(x[0]))

    # Write to CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "row"])
        writer.writerows(submission_rows)

    logger.info(f"Submission saved to {output_path}")