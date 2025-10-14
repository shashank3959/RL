# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint management utilities for HF models."""

import os
from typing import Any, Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from transformers import AutoConfig, AutoTokenizer


## modified from pytorch tutorial https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
class ModelState(Stateful):
    """Helper class for tracking model state in distributed checkpointing.

    This class is compliant with the Stateful protocol, allowing DCP to automatically
    call state_dict/load_state_dict as needed in the dcp.save/load APIs.

    Args:
        model: The PyTorch model to track.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        """Get the model's state dictionary.

        Returns:
            dict: Dictionary containing the model's state dict with CPU offloading enabled.
        """
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict = get_model_state_dict(
            self.model,
            options=torch.distributed.checkpoint.state_dict.StateDictOptions(
                cpu_offload=True
            ),
        )
        return model_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dictionary into the model.

        Args:
            state_dict (dict): State dictionary to load.
        """
        # sets our state dicts on the model, now that we've loaded
        set_model_state_dict(
            self.model,
            state_dict,
        )


class OptimizerState(Stateful):
    """Helper class for tracking optimizer state in distributed checkpointing.

    This class is compliant with the Stateful protocol, allowing DCP to automatically
    call state_dict/load_state_dict as needed in the dcp.save/load APIs.

    Args:
        model: The PyTorch model associated with the optimizer.
        optimizer: The optimizer to track.
        scheduler: Optional learning rate scheduler.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self) -> dict[str, Any]:
        """Get the optimizer and scheduler state dictionaries.

        Returns:
            dict: Dictionary containing the optimizer and scheduler state dicts with CPU offloading enabled.
        """
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        optimizer_state_dict = get_optimizer_state_dict(
            self.model,
            self.optimizer,
            options=torch.distributed.checkpoint.state_dict.StateDictOptions(
                cpu_offload=True
            ),
        )

        state_dict = {
            "optim": optimizer_state_dict,
        }
        if self.scheduler is not None:
            state_dict["sched"] = self.scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dictionaries into the optimizer and scheduler.

        Args:
            state_dict (dict): State dictionary containing optimizer and scheduler states to load.
        """
        # sets our state dicts on the optimizer, now that we've loaded
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            state_dict["optim"],
        )

        ## load the scheduler state if it exists
        if "sched" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["sched"])


from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_rl.utils.automodel_checkpoint import (
    load_checkpoint,
    save_checkpoint,
)

def load_model_from_checkpoint(
    checkpoint_path: str,
    base_model_path: Optional[str] = None,
) -> NeMoAutoModelForCausalLM:
    """Load a VLM model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory
        base_model_path: Path to the base model checkpoint. This can either be something like 'google/gemma-3-4b-it' or a local path to the base model. This is not required if restoring from a consolidated HF checkpoint.
        use_liger_kernel: Whether to use Liger kernel optimizations

    Returns:
        Loaded NeMoAutoModelForCausalLM model
    """
    # initialize distributed
    device = "cpu"
    if base_model_path is None:
        raise ValueError("base_model_path is required if not restoring from a consolidated HF checkpoint.")

    model = NeMoAutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        use_liger_kernel=False,
        trust_remote_code=True,
    ).to(device)

    load_checkpoint(
        model=model,
        weights_path=checkpoint_path,
    )
    print(f"✅ Model loaded successfully from {checkpoint_path}")
    return model

def convert_dcp_to_hf(
    dcp_ckpt_path: str,
    hf_ckpt_path: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    overwrite: bool = False,
) -> str:
    """Convert a Torch DCP checkpoint to a Hugging Face checkpoint.

    This is not an optimized utility. If checkpoint is too large, consider saving DCP during training
    and using this utility to convert to HF format.

    Args:
        dcp_ckpt_path (str): Path to DCP checkpoint
        hf_ckpt_path (str): Path to save HF checkpoint
        model_name_or_path (str): Model name or path for config
        tokenizer_name_or_path (str, optional): Tokenizer name or path.
                                               Defaults to model_name_or_path if None.
        overwrite (bool, optional): Whether to overwrite existing checkpoint. Defaults to False.

    Returns:
        str: Path to the saved HF checkpoint

    Raises:
        FileExistsError: If HF checkpoint already exists and overwrite is False
    """
    if os.path.exists(hf_ckpt_path) and not overwrite:
        raise FileExistsError(
            f"HF checkpoint already exists at {hf_ckpt_path}. Delete it to run or set overwrite=True."
        )

    os.makedirs(hf_ckpt_path, exist_ok=True)

    model = load_model_from_checkpoint(dcp_ckpt_path, model_name_or_path)

    save_checkpoint(
        model=model,
        weights_path=hf_ckpt_path,
        model_state_dict_keys=model.state_dict().keys(),
        model_save_format="safetensors",
        save_consolidated=True
    )
    
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.save_pretrained(hf_ckpt_path)

    # TODO: After the following PR gets merged:
    # https://github.com/NVIDIA-NeMo/RL/pull/148/files
    # tokenizer should be copied from policy/tokenizer/* instead of relying on the model name
    # We can expose a arg at the top level --tokenizer_path to plumb that through.
    # This is more stable than relying on the current NeMo-RL get_tokenizer() which can
    # change release to release.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=True
    )
    tokenizer.save_pretrained(hf_ckpt_path)
    return hf_ckpt_path
