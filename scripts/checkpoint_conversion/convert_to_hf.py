# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
import torchtitan.protocols.train_spec as train_spec_module
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import ConfigManager


@torch.inference_mode()
def convert_to_hf(input_dir, output_dir, model_name, model_flavor, config):
    # load model and model args so that we can get the state dict shape
    train_spec = train_spec_module.get_train_spec(model_name)
    model_args = train_spec.model_args[model_flavor]
    model_args.update_from_config(config)

    with torch.device("cpu"):
        model = train_spec.model_cls(model_args)
    model = ModelWrapper(model)

    sd_adapter = train_spec.state_dict_adapter(model_args)
    assert (
        sd_adapter is not None
    ), "trying to convert checkpoint from DCP to HF safetensors format, but sd_adapter is not provided."

    print("Getting statedict...")
    # allocate state dict memory with empty weights to load checkpoint
    state_dict = model._get_state_dict()

    print("Start loading checkpoint...")
    dcp.load(
        state_dict,
        checkpoint_id=input_dir,
    )

    print("Checkpoint loaded.")

    # convert state dict tt->hf
    hf_state_dict = sd_adapter.to_hf(state_dict)

    fqn_to_index_mapping = {}
    num_fqns_per_file = 30

    for i, key in enumerate(hf_state_dict.keys()):
        group_num = (i // num_fqns_per_file) + 1
        fqn_to_index_mapping[key] = group_num

    storage_writer = HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )

    print("Saving HF checkpoint...")
    dcp.save(
        hf_state_dict,
        storage_writer=storage_writer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCP weights to HF format.")
    parser.add_argument(
        "input_dir", type=Path, help="Input directory with DCP weights."
    )
    parser.add_argument(
        "output_dir", type=Path, help="Output directory for HF checkpoint."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="TOML config file path (required)"
    )
    args = parser.parse_args()
    
    # Load configuration from toml file
    config_manager = ConfigManager()
    config = config_manager.parse_args([f"--job.config_file={args.config}"])

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        config.model.name,
        config.model.flavor,
        config
    )
