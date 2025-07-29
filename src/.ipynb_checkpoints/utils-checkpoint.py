import csv

import json

import matplotlib.pyplot as plt

import numpy as np

import os.path as osp

import pandas as pd

import time
from tqdm import tqdm 
from typing import List, Tuple

# Load the competition files
def load_data(data_path:str):
    """Load training and test data"""
    # Load training data (includes solutions for learning)
    with open(osp.join(data_path, 'train.json'), 'r') as f:
        train_data = json.load(f)

    # Load test data (no solutions)
    with open(osp.join(data_path, 'test.json'), 'r') as f:
        test_data = json.load(f)

    return train_data, test_data

def plot_grid(grid, ax, title):
    """Plot a single grid"""
    # Color palette for ARC (0-9)
    colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title)
    ax.grid(True, which='both', color='lightgray', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def visualize_problem(problem_data, problem_id, max_examples=3):
    """Visualize a complete problem"""
    n_train = min(len(problem_data['train']), max_examples)
    n_test = len(problem_data['test'])

    fig, axes = plt.subplots(n_train + n_test, 2, figsize=(6, 4*(n_train+n_test)))
    if n_train + n_test == 1:
        axes = axes.reshape(1, -1)

    # Plot training examples
    for i in range(n_train):
        plot_grid(problem_data['train'][i]['input'], axes[i, 0], f'Train {i+1} Input')
        plot_grid(problem_data['train'][i]['output'], axes[i, 1], f'Train {i+1} Output')

    # Plot test examples
    for i in range(n_test):
        row = n_train + i
        plot_grid(problem_data['test'][i]['input'], axes[row, 0], f'Test {i+1} Input')

        # For test, we need to predict the output
        if 'output' in problem_data['test'][i]:
            # Training data has solutions
            plot_grid(problem_data['test'][i]['output'], axes[row, 1], f'Test {i+1} Output')
        else:
            # Test data - no solution
            axes[row, 1].text(0.5, 0.5, '?', ha='center', va='center', fontsize=50)
            axes[row, 1].set_title(f'Test {i+1} Output (Predict this!)')
            axes[row, 1].axis('off')

    plt.suptitle(f'Problem: {problem_id}', fontsize=16)
    plt.tight_layout()
    plt.show()

def analyze_transformation(input_grid, output_grid):
    """Analyze the transformation between input and output"""
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)

    analysis = {
        'size_change': {
            'input_shape': input_arr.shape,
            'output_shape': output_arr.shape,
            'same_size': input_arr.shape == output_arr.shape
        },
        'colors': {
            'input_colors': sorted(list(set(input_arr.flatten()))),
            'output_colors': sorted(list(set(output_arr.flatten()))),
        }
    }

    # Check for simple transformations
    if analysis['size_change']['same_size']:
        # Check rotations
        if np.array_equal(output_arr, np.rot90(input_arr, 1)):
            analysis['transformation'] = 'rotate_90'
        elif np.array_equal(output_arr, np.rot90(input_arr, 2)):
            analysis['transformation'] = 'rotate_180'
        elif np.array_equal(output_arr, np.rot90(input_arr, 3)):
            analysis['transformation'] = 'rotate_270'
        elif np.array_equal(output_arr, np.fliplr(input_arr)):
            analysis['transformation'] = 'flip_horizontal'
        elif np.array_equal(output_arr, np.flipud(input_arr)):
            analysis['transformation'] = 'flip_vertical'
        else:
            analysis['transformation'] = 'complex'
    else:
        analysis['transformation'] = 'size_change'

    return analysis

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

        return grid if grid else [[0]]
    except Exception as e:
        print(f"Error parsing grid: {e}")
        return [[0]]
    
def evaluate_on_training(model, train_data, num_examples=5):
    """Evaluate model on training examples"""
    correct = 0
    total = 0

    print(f"\n🧪 Testing on {num_examples} training examples...")

    for i, (problem_id, problem) in enumerate(train_data.items()):
        if i >= num_examples:
            break

        print(f"\nProblem {i+1}/{num_examples} (ID: {problem_id}):")

        # Create test version
        test_version = {
            'train': problem['train'],
            'test': [{'input': problem['test'][0]['input']}]
        }

        # Solve
        start_time = time.time()
        prediction = model.solve(test_version)
        solve_time = time.time() - start_time

        # Check if correct
        expected = problem['test'][0]['output']
        is_correct = (prediction == expected)

        if is_correct:
            correct += 1
        total += 1

        print(f"  Time: {solve_time:.2f}s")
        print(f"  Correct: {'✅' if is_correct else '❌'}")

        # Show shapes
        try:
            print(f"  Expected shape: {len(expected)}x{len(expected[0])}")
            print(f"  Predicted shape: {len(prediction)}x{len(prediction[0])}")
        except:
            print(f"  Shape error in prediction")

    accuracy = correct / total if total > 0 else 0
    print(f"\n📊 Accuracy: {accuracy:.1%} ({correct}/{total})")
    return accuracy

def create_submission(
    test_data, 
    model, 
    output_file='submission.json', 
    csv_file='submission.csv'
    ):
    """Create submission files in both JSON and CSV formats"""
    submission_json = {}
    submission_csv = []
    total = len(test_data)

    print(f"\n📝 Generating submission for {total} test problems...")

    for i, (problem_id, problem) in enumerate(test_data.items()):
        # Progress update
        # if (i + 1) % 20 == 0:
        #     print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

        try:
            # Check how many test cases this problem has
            num_test_cases = len(problem['test'])
            predictions = []

            # Solve each test case
            for test_idx in range(num_test_cases):
                # Create a version with just this test case
                single_test_problem = {
                    'train': problem['train'],
                    'test': [problem['test'][test_idx]]
                }

                # Solve this test case
                prediction = model.solve(single_test_problem)

                if isinstance(prediction, list) and len(prediction) > 0:
                    predictions.append(prediction)
                else:
                    # Fallback to input
                    predictions.append(problem['test'][test_idx]['input'])

            # JSON format stores all predictions
            submission_json[problem_id] = predictions

            # CSV format: each matrix and each row gets its own CSV row
            for matrix_idx, prediction in enumerate(predictions):
                matrix_num = matrix_idx + 1  # Start from 1

                for row_idx, row in enumerate(prediction):
                    row_num = row_idx + 1  # Start from 1
                    ""
                    # Convert row values to string
                    row_string = ''.join(str(cell) for cell in row)

                    # Create ID with matrix and row indices
                    if num_test_cases > 1:
                        # Multiple test cases: include matrix index
                        csv_id = f"{problem_id}_{matrix_num}_{row_num}"
                    else:
                        # Single test case: just row index
                        csv_id = f"{problem_id}_{row_num}"

                    submission_csv.append({
                        'ID': csv_id,
                        'row': row_string
                    })

        except Exception as e:
            print(f"  Error on {problem_id}: {e}")
            # Fallback: submit inputs as outputs
            fallback_predictions = []
            for test_case in problem['test']:
                fallback_predictions.append(test_case['input'])

            submission_json[problem_id] = fallback_predictions

            # Create CSV rows for fallback
            for matrix_idx, input_grid in enumerate(fallback_predictions):
                matrix_num = matrix_idx + 1
                for row_idx, row in enumerate(input_grid):
                    row_num = row_idx + 1
                    row_string = ''.join(str(cell) for cell in row)

                    if len(fallback_predictions) > 1:
                        csv_id = f"{problem_id}_{matrix_num}_{row_num}"
                    else:
                        csv_id = f"{problem_id}_{row_num}"

                    submission_csv.append({
                        'ID': csv_id,
                        'row': row_string
                    })

    # Save JSON submission
    with open(output_file, 'w') as f:
        json.dump(submission_json, f, separators=(',', ':'))

    # Save CSV submission
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'row'])
        writer.writeheader()
        writer.writerows(submission_csv)

    print(f"\n Submissions saved:")
    print(f"   JSON format: {output_file}")
    print(f"   CSV format: {csv_file}")
    print(f"   Total problems: {len(submission_json)}")
    print(f"   Total CSV rows: {len(submission_csv)}")

    # # Show sample of CSV format
    # if submission_csv:
    #     print(f"\nSample CSV entries (first 10 rows):")
    #     for i in range(min(10, len(submission_csv))):
    #         print(f"   {submission_csv[i]['ID']}: {submission_csv[i]['row']}")

    # Show example of multi-test problem
    multi_test_problems = [p for p in test_data.items() if len(p[1]['test']) > 1]
    if multi_test_problems:
        example_id, example_prob = multi_test_problems[0]
        print(f"\n Example multi-test problem '{example_id}':")
        print(f"   Number of test cases: {len(example_prob['test'])}")
        if example_id in submission_json:
            for i, pred in enumerate(submission_json[example_id]):
                print(f"   Matrix {i+1}: {len(pred)}x{len(pred[0])} ({len(pred)} rows)")

    return submission_json, submission_csv

# Additional utility functions that might be helpful
def grid_similarity(grid1: np.ndarray, grid2: np.ndarray) -> float:
    """
    Calculate similarity between two grids.
    """
    if grid1.shape != grid2.shape:
        return 0.0
    
    matches = np.sum(grid1 == grid2)
    total = grid1.size
    return matches / total if total > 0 else 0.0

def find_repeating_patterns(grid: np.ndarray, pattern_size: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find positions of repeating patterns in a grid.
    """
    rows, cols = pattern_size
    positions = []
    
    if grid.shape[0] >= rows and grid.shape[1] >= cols:
        for i in range(grid.shape[0] - rows + 1):
            for j in range(grid.shape[1] - cols + 1):
                positions.append((i, j))
    
    return positions

