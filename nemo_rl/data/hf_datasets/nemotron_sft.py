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

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from datasets import (
    Dataset as HFDataset,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
)

from nemo_rl.data.interfaces import TaskDataSpec


def _format_nemotron_sft_row(example: dict[str, Any]) -> dict[str, Any]:
    """Format a single Nemotron example to internal message schema.

    Supports the new dataset schema from
    "nvidia/Nemotron-Post-Training-Dataset-v1" where each row typically
    contains a "messages" list, and also gracefully handles legacy rows that
    may provide "input"/"output" fields.
    """
    # # Preferred path: pass through messages if present
    # if isinstance(example.get("messages"), list) and len(example["messages"]) > 0:
    #     # Ensure each message has only role/content keys to align with our internal schema
    #     normalized: list[dict[str, str]] = []
    #     for msg in example["messages"]:
    #         # Whitelist only the keys we allow; drop e.g. tool_calls, tool_call_id, name, etc.
    #         role = str(msg.get("role", "user"))
    #         content = msg.get("content")
    #         if content is None:
    #             content = ""
    #         content_str = content if isinstance(content, str) else str(content)
    #         normalized.append({"role": role, "content": content_str})

    return {
        "messages": example["messages"],
        "task_name": "nemotron_sft",
    }


def _load_raw_nemotron_sft(
    subsets: Union[str, Sequence[str]] = ("math",),
    cache_dir: Optional[str] = None,
) -> Any:
    """Load category splits from Nemotron-Post-Training-Dataset-v1.

    Args:
        subsets: One or more category splits to load (e.g., "chat", "code", "math", "stem", "tool_calling").
        cache_dir: Optional HF datasets cache directory.

    Returns:
        A single Dataset containing the concatenation of the requested subsets.
    """
    if isinstance(subsets, str):
        subsets = [subsets]

    datasets: list[Any] = []
    for subset in subsets:
        ds = load_dataset(
            "nvidia/Nemotron-Post-Training-Dataset-v1",
            split=subset,
            cache_dir=cache_dir,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def prepare_nemotron_sft_dataset(
    subsets: Union[str, Sequence[str]] = ("math",),
    seed: int = 42,
    test_size: float = 0.05,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> dict[str, Any | None]:
    """Load and split the Nemotron Post-Training dataset into train/validation sets.

    Uses Hugging Face's train_test_split for a quick ephemeral split. For
    reproducible experiments, consider preprocessing once and persisting splits.
    """
    print(
        "WARNING: For reproducible experiments, preprocess the dataset once and define your own HfDataset subclass that directly uses the preprocessed datasets."
    )

    original_ds = _load_raw_nemotron_sft(subsets=subsets, cache_dir=cache_dir)

    if max_samples is not None and max_samples > 0:
        original_ds = original_ds.shuffle(seed=seed).select(range(min(max_samples, len(original_ds))))

    split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)

    train_formatted = split_ds["train"].map(
        _format_nemotron_sft_row,
        remove_columns=split_ds["train"].column_names,
    )
    val_formatted = split_ds["test"].map(
        _format_nemotron_sft_row,
        remove_columns=split_ds["test"].column_names,
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class NemotronSFTDataset:
    def __init__(
        self,
        subsets: Union[str, Sequence[str]] = ("math",),
        seed: int = 42,
        test_size: float = 0.05,
        prompt_file: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """Nemotron SFT HF dataset wrapper producing train/validation and TaskDataSpec.

        Args:
            subsets: One or more of {"chat", "code", "math", "stem", "tool_calling"}.
            seed: Random seed for splitting.
            test_size: Fraction for validation split.
            prompt_file: Optional prompt file path to be applied via TaskDataSpec.
        """
        self.formatted_ds = prepare_nemotron_sft_dataset(
            subsets=subsets,
            seed=seed,
            test_size=test_size,
            cache_dir=cache_dir,
            max_samples=max_samples,
        )

        self.task_spec = TaskDataSpec(
            task_name="Nemotron-Post-Training-Dataset-v1",
            prompt_file=prompt_file,
        )


