import ast

from collections import Counter

import gc 

from huggingface_hub import login

from src.__init__ import logger
from src.config import Config
try:
    from mlx_lm import load, generate
except ImportError:
    pass
import numpy as np

from pprint import pprint
import re

import time
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig, PreTrainedTokenizer
from typing import Tuple

from src.utils import infer_out_shape, get_grid_shape, text_to_grid, fill_irregular_grid

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
            target_platform:str=Config.target_platform
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

            self.llm = self.llm.eval()

    def load_llm(
            self,
            target_platform:str=Config.target_platform,
            base_llm_name:str = Config.base_llm
        )->Tuple:

        if self.verbose:
            logger.info(f"{target_platform=}")
            logger.info(f"{base_llm_name=}")
        
        if Config.HF_TOKEN:
            login(Config.HF_TOKEN)
            
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
            self.tokenizer  = AutoTokenizer.from_pretrained(base_llm_name)
            self.llm      = AutoModelForCausalLM.from_pretrained(
                base_llm_name, 
                torch_dtype=Config.dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code   = Config.trust_remote_code,
            )

        logger.info(f"✅ Successfully loaded base LLM ({self.base_llm_name}).")

    def prepare_model_inputs(
            self,
            prompt:str, 
            expected_num_rows:int,
            task_data=None,
            target_platform:str=Config.target_platform,
            verbose:bool=True
        ):

        if verbose==True:
            logger.info("Preparing inputs")

        if task_data is not None:
            train_pairs = task_data["train"]
            test = task_data["test"][0]
            
            expected_out_shape = infer_out_shape(
                    train_pairs=train_pairs, 
                    test_input=test["input"],  
                    expected_rows=expected_num_rows,
                    verbose=False
                )
            
            prompt += "Please only return the most likely test output grid in the same format as the input and based on your understanding of the example puzzles.\n\n"
            prompt += "Make shure to use the most common value in the test input if you find it difficult to infer it from your analysis\n"
            prompt += "Do not provide explanations after generating the output.\n\n"
            prompt += f"No need to think too much but carefully analyze the examples. Here are the training examples: {train_pairs}\n\n"
            prompt += f"And here is the problem for you to solve. Keep the output shape as {expected_out_shape}: {test["input"]}\n"
   


        messages = [
            {"role": "system","content": self.base_prompt},
            {"role": "user", "content": prompt}
        ]
        if target_platform=="mlx":
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            )
            inputs = {k: v.to(Config.device) for k, v in inputs.items() if k!="token_type_ids"}
        
        if verbose==True:
            logger.info("Done!")

        return inputs

    def get_response(
            self,
            inputs,
            max_tokens:int=Config.max_tokens,
            verbose:bool=True,
            target_platform:str=Config.target_platform
        ):
        
        if verbose==True:
            logger.info("Prompting LLM...")

        if target_platform=="mlx":
            response = generate(
                self.llm, 
                self.tokenizer, 
                prompt=inputs, 
                verbose=verbose,
                max_tokens=max_tokens
        )
        else:
            len_inp_prompt = inputs["input_ids"].shape[-1]
            
            generation_config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=Config.temperature,
                top_p=Config.top_p,
                do_sample=Config.do_sample,
                verbose=verbose,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                repetition_penalty=Config.repetition_penalty
            )
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    generation_config=generation_config,
                )

                response = self.tokenizer.batch_decode(outputs[:, len_inp_prompt:])[0] # batch_size=1
       
            if verbose==True:
                logger.info("Done!")
        
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

            if self.verbose:
                logger.info("Fallback solver...")
            results.append(self.fallback_grid(test_input["input"], inferred_out_shape))
            
        return results[0] # this will ensure we only return one prediction

    def copy_input_solver(self, test_input, expected_shape=None, verbose:bool=False):
        """
        Returns the input grid as the output grid, optionally adjusting to expected shape.
        
        Args:
            test_input (List[List[int]]): The input grid.
            expected_shape (tuple, optional): (rows, cols). If provided, the output is resized accordingly.
            
        Returns:
            List[List[int]]: Output grid.
        """
        if verbose:
            logger.info("Copy input solver...")  

        output = [row[:] for row in test_input]

        if expected_shape:
            output = self.adjust_rows_and_columns(grid=output, expected_shape=expected_shape)
            
        if self.verbose:
            logger.info("Done!")  

        return output

    def solve_with_llm(
        self, 
        task,
        task_id:str,
        expected_rows:int=None,
        verbose:bool=False
    ):
        
        train = task["train"]
        test = task["test"][0]
    
        expected_shape = infer_out_shape(
            train_pairs=train, 
            test_input=test["input"],  
            expected_rows=expected_rows,
            verbose=False
        )
        
        if verbose:
            logger.info(f"Expected shape: {expected_shape}")
        
        try:
            inputs = self.prepare_model_inputs(
                prompt=self.base_prompt,
                task_data=task,
                expected_num_rows=expected_shape[0] if expected_shape else None,
                target_platform=self.target_platform,
                verbose=False
            )
    
            start_time = time.time()
            
            if expected_shape[0] > Config.MAX_N_ROWS:
                out_grid = self.copy_input_solver(task["test"][0]["input"], expected_shape, verbose=verbose)
            else:
                response = self.get_response(inputs, verbose=verbose)
                extracted_grids = re.findall(r"\[\s*\[.*?\]\s*\]", response, re.DOTALL)
                
                if len(extracted_grids)==0:
                    out_grid = task["test"][0]["input"]
                else:
                    out_grid = extracted_grids[-1]

                out_grid = text_to_grid(out_grid)
            
            elapsed = time.time() - start_time
            
            logger.info(f"Task #{task_id} \t Completion time: {elapsed:.4f} seconds")

            out_grid = self.adjust_rows_and_columns(grid=out_grid, expected_shape=expected_shape)
    
            torch.cuda.empty_cache()
            gc.collect()
            
            
        except Exception as e:
            response = None
            out_grid = self.copy_input_solver(task["test"][0]["input"], expected_shape, verbose=verbose)
            out_grid = self.adjust_rows_and_columns(grid=out_grid, expected_shape=expected_shape)
            
            logger.error(task_id)
            logger.error(e)

        return response, out_grid
            

    def fallback_grid(self, test_input, expected_shape):
        return self.copy_input_solver(test_input, expected_shape)

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
            expected_shape
        ):
        """
        Adjusts a grid to the expected (rows, cols) shape by cropping or padding.
    
        Args:
            grid (List[List[int]]): The grid to adjust.
            expected_shape (tuple): (rows, cols)
    
        Returns:
            List[List[int]]: Resized grid.
        """
    
        flat = [cell for row in grid for cell in row]
        fill_value = Counter(flat).most_common(1)[0][0] if flat else Config.DEFAULT_BG_VALUE

        if self.verbose:
            logger.info("Adjusting grid...")
            logger.info(f"{expected_shape=}, {fill_value=}")
    
        exp_rows, exp_cols = expected_shape
        adjusted = []
    
        for i in range(exp_rows):
            if i < len(grid):
                row = grid[i][:exp_cols]
                if len(row) < exp_cols:
                    row += [fill_value] * (exp_cols - len(row))
            else:
                row = [fill_value] * exp_cols
            adjusted.append(row)
    
        if self.verbose:
            final_shape = get_grid_shape(adjusted)
            logger.info(f"{final_shape=}")
    
        return adjusted





