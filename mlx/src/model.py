import ast

from src.__init__ import logger
from src.config import Config

from mlx_lm import load, generate

import numpy as np

from pprint import pprint
import re

import time
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig, PreTrainedTokenizer, StoppingCriteria
from typing import Tuple

from src.utils import infer_out_shape, get_grid_shape, text_to_grid, grid_to_text


class RowLimitStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, expected_rows, row_token="["):
        self.tokenizer = tokenizer
        self.expected_rows = expected_rows
        self.row_token_id = tokenizer.encode(row_token, add_special_tokens=False)[0]
        self.row_count = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.row_count = list(input_ids[0].cpu().numpy()).count(self.row_token_id)
        return self.row_count >= self.expected_rows


class ARCModel:
    """
    Enhanced ARC solver with multiple reasoning strategies.
    """

    def __init__(
            self, 
            dtype:torch.dtype=Config.dtype,
            model_name:str=Config.model_name, 
            system_prompt_path="base_prompt.txt", 
            base_llm_name:str=Config.base_llm,
            verbose:bool=True,
            target_platform:str="mlx"
        ):
        self.model_name         = model_name
        self.verbose            = verbose
        self.base_llm_name      = base_llm_name
        self.base_prompt        = ""
        self.llm                = None
        self.tokenizer          = None
        self.target_platform    = target_platform

        if base_llm_name is not None:
            with open(system_prompt_path, "r") as f:
                self.base_prompt = f.read().strip()
            
            self.load_llm(
                target_platform=self.target_platform,
                base_llm_name=self.base_llm_name
            )


    def load_llm(
            self,
            target_platform:str="mlx",
            base_llm_name:str = Config.base_llm,
            quantize_model:bool=False
        )->Tuple:

        if self.verbose:
            logger.info(f"{target_platform=}")
            logger.info(f"{base_llm_name=}")

        if target_platform =="mlx":
            self.llm, self.tokenizer = load(
                base_llm_name,
                tokenizer_config={
                    "eos_token": "<|endoftext|>", 
                    "trust_remote_code": Config.trust_remote_code,
                    "temperature": Config.temperature,
                    "top_p": Config.top_p
                }
            )
            is_model_chat_ready = hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None
        
            assert is_model_chat_ready==True, "Model tokenizer is missing attr apply_chat_template"
        else:
            quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_use_double_quant=True
            )
            self.tokenizer  = AutoTokenizer.from_pretrained(base_llm_name)
            self.llm      = AutoModelForCausalLM.from_pretrained(
                base_llm_name, 
                torch_dtype=Config.dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code   = Config.trust_remote_code,
                quantization_config=quantization_config if quantize_model else None
            ).to(Config.device)

        logger.info(f"✅ Successfully loaded base LLM ({self.base_llm_name}).")

    def prepare_model_inputs(
            self,
            prompt:str, 
            expected_num_rows:int,
            task_data=None,
            target_platform:str="mlx"
        ):

        if task_data is not None:
            train_pairs = task_data["train"]
            test = task_data["test"][0]
            
            expected_out_shape = infer_out_shape(
                    train_pairs=train_pairs, 
                    test_input=test["input"],  
                    expected_rows=expected_num_rows,
                    verbose=False
                )

            prompt += f"Think but not too long. Here are the training examples: {train_pairs}\n"
            prompt += f"Return an output with exactly {expected_out_shape[0]} rows.\n"
            prompt += "Please, do not re-iterate the test input in your answer since it is unnessecary.\n"
            prompt += "Just apply the rules and return the most likely test output grid in the same format as the input.\n"
            prompt += "Also, providing too many explanations is optional.\n"
            prompt += f"Here is the problem for you to solve: {test["input"]}\n\n"


        messages = [
        {
            "role": "user", 
            "content": prompt
            }
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        if target_platform!="mlx":
            inputs = {k: v.to(Config.device) for k, v in inputs.items() if k!="token_type_ids"}

        return inputs

    def get_response(
            self,
            inputs,
            max_tokens:int=Config.max_tokens,
            verbose:bool=True,
            expected_rows:int=None,
            target_platform:str="mlx"
        ):
        
        if target_platform=="mlx":
            response = generate(
                self.llm, 
                self.tokenizer, 
                prompt=inputs, 
                verbose=verbose,
                max_tokens=max_tokens
        )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=Config.temperature,
                top_p=Config.top_p,
                do_sample=Config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                repetition_penalty=Config.repetition_penalty,
                stopping_criteria=[RowLimitStoppingCriteria(self.tokenizer, expected_rows)]
            )
            with torch.no_grad():
                try:
                    outputs = self.llm.generate(
                        **inputs.to(Config.device),
                        generation_config=generation_config,
                    )
                except:
                    inputs = {k: v.to(Config.device) for k, v in inputs.items() if k!="token_type_ids"}
                    outputs = self.llm.generate(
                        **inputs.to(Config.device),
                        generation_config=generation_config,
                    )
                    
                    outputs = self.tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
                
                response = outputs[0]

        return response

    def solve(self, task, expected_rows:int=None):
        train = task["train"]
        test = task["test"]
        results = []

        for test_input in test:
            if "output" in test_input.keys():
                # adjust expected row num
                expected_rows = len(test_input["output"])

            inferred_out_shape = infer_out_shape(
                train_pairs=train, 
                test_input=test_input["input"],  
                expected_rows=expected_rows,
                verbose=self.verbose
            )

            if self.base_llm_name is not None:
                if self.verbose:
                    logger.info("Solving with LLM...")
                try:
                    candidates = self.solve_with_llm(train, test_input["input"], inferred_out_shape[0])
                    results.append(candidates[0])
                except Exception as e:
                    logger.error(f"Failed llm execution: {e}")
            else:
                if self.verbose:
                    logger.info("Fallback solver...")
                results.append(self.copy_input_solver(test["input"], inferred_out_shape))
                
        return results[0] # this will ensure we only return one prediction

    def copy_input_solver(self, test_input, expected_shape=None):
        """
        Returns the input grid as the output grid, optionally adjusting to expected shape.
        
        Args:
            test_input (List[List[int]]): The input grid.
            expected_shape (tuple, optional): (rows, cols). If provided, the output is resized accordingly.
            
        Returns:
            List[List[int]]: Output grid.
        """
        if self.verbose:
            logger.info("Copy input solver...")  

        output = [row[:] for row in test_input]

        if expected_shape:
            output = self.adjust_rows_and_columns(grid=output, expected_shape=expected_shape)
            # output = self.adjust_columns_only(grid=output, expected_cols=expected_shape[1])
            
        if self.verbose:
            logger.info("Done!")  

        return output

    def solve_with_llm(self, task, expected_shape, verbose:bool=False):

        if verbose:
            logger.info(f"Expected shape: {expected_shape}")
        
        inputs = self.prepare_model_inputs(
            prompt=self.base_prompt,
            task_data=task,
            expected_num_rows=expected_shape[0],
            target_platform=self.target_platform
        )

        start_time = time.time()
        if expected_shape[0] > Config.MAX_N_ROWS:
            out_grid = self.copy_input_solver(task["test"][0]["input"], expected_shape)
        else:
            response = self.get_response(inputs, verbose=verbose, expected_rows=expected_shape[0])
            extracted_grids = re.findall(r"\[\s*\[.*?\]\s*\]", response, re.DOTALL)
            
            if len(extracted_grids)==0:
                out_grid = task["test"][0]["input"]
            else:
                out_grid =             extracted_grids[-1]


            out_grid = text_to_grid(out_grid)
            
        elapsed = time.time() - start_time
        if self.verbose:
            logger.info(f"Completion time: {elapsed:.4f} seconds")
        
        if expected_shape:
            output = self.adjust_rows_and_columns(grid=out_grid, expected_shape=expected_shape)
            
        return output, elapsed

    def pixel_accuracy(self, pred, target):
        try:
            pred_arr = np.array(pred)
            target_arr = np.array(target)

            if pred_arr.shape != target_arr.shape:
                return 0.0

            total = pred_arr.size
            match = np.sum(pred_arr == target_arr)
            return float(match / total) if total > 0 else 0.0
        except:
            return 0.0
  
    def evaluate(self, task, predictions):
        """
        Compares predicted outputs against ground truth outputs in the task.
        Returns: dict with accuracy metrics
        """
        assert "test" in task and all("output" in t for t in task["test"]), "Missing test outputs"

        correct = 0
        total = len(task["test"])
        pixel_scores = []

        for pred, test_case in zip(predictions, task["test"]):
            target = test_case["output"]

            if pred == target:
                correct += 1
                pixel_scores.append(1.0)
            else:
                match = self.pixel_accuracy(pred, target)
                pixel_scores.append(match)

        return {
            "exact_match": correct / total,
            "mean_pixel_accuracy": sum(pixel_scores) / total
        }

    def adjust_rows_and_columns(
            self, 
            grid, 
            expected_shape, 
            fill_value=Config.DEFAULT_BG_VALUE
        ):
        """
        Adjusts a grid to the expected (rows, cols) shape by cropping or padding.
        
        Args:
            grid (List[List[int]]): The grid to adjust.
            expected_shape (tuple): (rows, cols)
            fill_value (int): Value to pad with, default is 7 (orange).
            
        Returns:
            List[List[int]]: Resized grid.
        """
        if self.verbose:
            logger.info("Adjusting grid...")  
            logger.info(f"{expected_shape=}")

        exp_rows, exp_cols = expected_shape
        adjusted = []

        for i in range(exp_rows):
            if i < len(grid):
                row = grid[i][:exp_cols]  # crop if needed
                if len(row) < exp_cols:
                    row += [fill_value] * (exp_cols - len(row)) 
            else:
                row = [fill_value] * exp_cols # pad grid
            adjusted.append(row)
       
        if self.verbose:
            final_shape = get_grid_shape(adjusted)
            logger.info(f"{final_shape=}")
        
        return adjusted




