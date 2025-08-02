import ast

from src.__init__ import logger
from src.config import Config

try:
    from mlx_lm import load, generate
except ImportError:
    pass
import numpy as np

import re

import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig, PreTrainedTokenizer
from typing import Tuple

from src.utils import infer_out_shape, get_grid_shape

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
            quantize_model:bool=Config.quantize_model
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
            self.tokenizer  = AutoTokenizer.from_pretrained(base_llm_name)
            
            if quantize_model:
                to_4_bit = Config.to_4_bit
                to_8_bit = False if to_4_bit==True else True                

                quantization_config = BitsAndBytesConfig(
                        load_in_4bit=to_4_bit,
                        load_in_8_bit=to_8_bit,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype="float16",
                        bnb_4bit_use_double_quant=True
                )
                                
                self.llm      = AutoModelForCausalLM.from_pretrained(
                    base_llm_name, 
                    torch_dtype=Config.dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code   = Config.trust_remote_code,
                    quantization_config=quantization_config
                )
                
                logger.info(f"Quantized version loaded (8-bit: {to_8_bit}, 4-bit: {to_4_bit})")                
            
            else:
                self.llm      = AutoModelForCausalLM.from_pretrained(
                    base_llm_name, 
                    torch_dtype=Config.dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code   = Config.trust_remote_code
                ).to(Config.device)

        logger.info(f"Successfully loaded base LLM ({self.base_llm_name}).")

    def prepare_model_inputs(
            self,
            prompt:str, 
            expected_num_rows:int=None,
            task_data=None,
            target_platform:str=Config.target_platform
        ):

        if task_data is not None:
            train_pairs = task_data["train"]
            
            test = task_data["test"][0]
            test_input = test["input"]
            
            expected_out_shape = infer_out_shape(
                    train_pairs=train_pairs, 
                    test_input=test_input,  
                    expected_rows=expected_num_rows,
                    verbose=False
                )

            prompt += f"Think but not too long. Here are the training examples: {train_pairs}\n"
            prompt += f"Return an output with exactly {expected_out_shape[0]} rows.\n"
            prompt += "Please, do not re-iterate the test input in your answer since it is unnessecary.\n"
            prompt += "Just apply the rules and return the most likely test output grid in the same format as the input.\n"
            prompt += "Also, providing too many explanations is optional.\n"
            prompt += f"Here is the problem for you to solve: {test_input}\n\n"


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
                max_new_tokens=Config.max_tokens,
                temperature=Config.temperature,
                top_p=Config.top_p,
                do_sample=Config.do_sample,
                verbose=True,
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

    def solve(
            self, 
            task, 
            task_id:str,
            expected_rows:int=None,
            verbose:bool=False
        ):
        train = task["train"]
        test = task["test"][0]

        if "output" in test.keys():
            # adjust expected row num
            expected_rows = len(test["output"])

        inferred_out_shape = infer_out_shape(
            train_pairs=train, 
            test_input=test["input"],  
            expected_rows=expected_rows,
            verbose=verbose
        )

        if self.base_llm_name is not None:
            if verbose:
                logger.info("Solving with LLM...")
            try:
                out_grid = self.solve_with_llm(
                    task=task, 
                    task_id=task_id,
                    expected_shape=inferred_out_shape,
                    verbose=verbose
                )
            except Exception as e:
                logger.error(f"Failed LLM execution: {e}")
        else:
            out_grid = self.copy_input_solver(
                test["input"], 
                inferred_out_shape,
                verbose=verbose
            )
                
        return out_grid

    def copy_input_solver(
            self, 
            test_input, 
            expected_shape=None,
            verbose:bool=False
            ):
        """
        Returns the input grid as the output grid, optionally adjusting to expected shape.
        
        Args:
            test_input (List[List[int]]): The input grid.
            expected_shape (tuple, optional): (rows, cols). If provided, the output is resized accordingly.
            
        Returns:
            List[List[int]]: Output grid.
        """
        if verbose:
            logger.info("Fallback - Copy input solver...")  

        output = [row[:] for row in test_input]

        if expected_shape:
            output = self.adjust_rows_and_columns(
                grid=output, 
                expected_shape=expected_shape,
                verbose=verbose
            )
            
        if verbose:
            logger.info("Done!")  

        return output

    def solve_with_llm(
            self, 
            task, 
            expected_shape,
            task_id:str=None,
            verbose:bool=False
        ):

        test_input = task["test"][0]["input"]
        test_in_shape = get_grid_shape(test_input)

        if verbose:
            logger.info(f"In. shape: \t\t{test_in_shape}")
            logger.info(f"Expected Out. shape: \t{expected_shape}")
        
        inputs = self.prepare_model_inputs(
            prompt=self.base_prompt,
            task_data=task,
            expected_num_rows=expected_shape[0],
            target_platform=self.target_platform
        )
        max_expected = max(expected_shape)
        start_time = time.time()
        if max_expected > Config.MAX_N:
            out_grid = self.copy_input_solver(test_input, expected_shape)
        else:
            response = self.get_response(
                inputs, 
                verbose=verbose
            )
            out_grid = self.extract_out_grid(response)
            
            if len(out_grid)==0:
                out_grid = task["test"][0]["input"]

        out_grid_shape = get_grid_shape(out_grid)
            
        elapsed = time.time() - start_time
        
        if task_id:
            logger.info(f"Task #{task_id} \t Completion time: {elapsed:.4f} seconds")
        
        if expected_shape and out_grid_shape!=expected_shape:
            out_grid = self.adjust_rows_and_columns(
                grid=out_grid, 
                expected_shape=expected_shape
            )
            
        return out_grid

    def extract_out_grid(self, response):
        # Find all list-of-lists patterns (non-greedy, multiline)
        matches = re.findall(r"\[\s*\[.*?\]\s*\]", response, re.DOTALL)
        output_grid = []
        
        if matches:
            try:
                output_grid = ast.literal_eval(matches[-1])
            except Exception as e:
                logger.error(e)
                raise e
            
        return output_grid
    
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
            fill_value=Config.DEFAULT_BG_VALUE,
            verbose:bool=False
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
        if verbose:
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
       
        if verbose:
            final_shape = get_grid_shape(adjusted)
            logger.info(f"{final_shape=}")
        
        return adjusted
