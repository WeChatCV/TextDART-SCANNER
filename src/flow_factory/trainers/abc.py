# Copyright 2026 Jayce-Ping
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

import json

# src/flow_factory/trainers/abc.py
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..advantage import AdvantageProcessor
from ..data_utils.loader import get_dataloader
from ..hparams import Arguments
from ..logger import LogFormatter, load_logger
from ..models.abc import BaseAdapter
from ..rewards import BaseRewardModel, MultiRewardLoader, RewardBuffer, RewardProcessor
from ..samples import BaseSample
from ..utils.image import standardize_image_batch
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class BaseTrainer(ABC):
    """
    Abstract Base Class for Flow-Factory trainers.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        config: Arguments,
        adapter: BaseAdapter,
    ):
        self.accelerator = accelerator
        self.config = config
        self.log_args = config.log_args
        self.model_args = config.model_args

        self.training_args = config.training_args
        self.eval_args = config.eval_args

        self.reward_args = config.reward_args
        self.eval_reward_args = (
            config.eval_reward_args or config.reward_args
        )  # If `eval_reward_args` is not given, use `reward_args`

        self.adapter = adapter
        self.epoch = 0
        self.step = 0
        self._cumulative_prompts = 0
        self._cumulative_train_time = 0.0

        # Restore epoch/step counters when continuing a previous run so that
        # logging, checkpoint naming, and seeding line up with the resumed
        # checkpoint (must run before `_init_logging_backend` resumes wandb).
        if self.log_args.resume_run and self.model_args.resume_path:
            self._restore_training_progress()

        self._initialization()
        self.adapter.post_init()
        self._init_logging_backend()

        self._patch_deepspeed_autocast(accelerator)
        self.autocast = partial(
            torch.autocast,
            device_type=accelerator.device.type,
            dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16,
        )

        if self.accelerator.is_local_main_process:
            self.adapter.log_trainable_parameters()

    @property
    def show_progress_bar(self) -> bool:
        """Whether to show tqdm progress bars."""
        return self.log_args.verbose and self.accelerator.is_local_main_process

    def should_continue_training(self) -> bool:
        """Outer epoch loop: continue unless a finite ``max_epochs`` has been reached."""
        m = self.training_args.max_epochs
        if m is None or m < 0:
            return True
        return self.epoch < m

    def log_data(self, data: Dict[str, Any], step: int):
        """Log data using the initialized logger."""
        if self.logger is not None:
            self.logger.log_data(data, step=step)

        # Print summary to console
        if self.accelerator.is_local_main_process:
            metrics = {
                k: v
                for k, v in ((k, LogFormatter.to_scalar(v)) for k, v in data.items())
                if v is not None
            }
            if metrics:
                parts = [f"[Step {step:04d} | Epoch {self.epoch:03d}]"]
                parts.extend(
                    (
                        f"{k}={int(v)}"
                        if isinstance(v, int) or (isinstance(v, float) and v.is_integer())
                        else f"{k}={v:.4f}"
                    )
                    for k, v in metrics.items()
                )
                logger.info(" ".join(parts))

    def _log_train_epoch_stats(self, num_local_samples: int, elapsed_seconds: float) -> None:
        """Update and log cumulative training statistics after each epoch.

        Args:
            num_local_samples: Number of samples generated on this GPU in this epoch.
            elapsed_seconds: Wall-clock time for this epoch's train portion (excl. eval).
        """
        world_size = self.accelerator.num_processes
        group_size = self.training_args.group_size
        epoch_prompts = num_local_samples * world_size // group_size
        self._cumulative_prompts += epoch_prompts
        self._cumulative_train_time += elapsed_seconds
        if self.accelerator.is_main_process:
            self.log_data(
                {
                    "train/total_prompts": self._cumulative_prompts,
                    "train/train_time": self._cumulative_train_time,
                },
                step=self.step,
            )

    def _log_eval_stats(self, total_eval_samples: int, elapsed_seconds: float) -> None:
        """Log per-evaluation statistics.

        Args:
            total_eval_samples: Global number of eval samples (across all GPUs).
            elapsed_seconds: Wall-clock time for this evaluation.
        """
        if self.accelerator.is_main_process:
            self.log_data(
                {
                    "eval/num_samples": total_eval_samples,
                    "eval/eval_time": elapsed_seconds,
                },
                step=self.step,
            )

    def _textdart_reward_names(self, split: str) -> List[str]:
        """Unique ``textdart`` reward names for ``split`` logging."""
        if split == "eval":
            multi = self.config.eval_reward_args or self.config.reward_args
        elif split == "train":
            multi = self.config.reward_args
        else:
            raise ValueError(f"split must be 'eval' or 'train', got {split!r}")
        if not multi:
            return []
        names: List[str] = []
        seen = set()
        for reward in multi:
            raw = str(getattr(reward, "reward_model", "") or "").strip().lower()
            if raw != "textdart" and not raw.endswith("textdartrewardmodel"):
                continue
            name = reward.name
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    def _gather_per_sample_extra_floats(
        self, samples: List[BaseSample], extra_key: str
    ) -> Optional[Any]:
        """All-gather a per-sample float extra, or ``None`` if any rank is incomplete."""
        values: List[float] = []
        complete = True
        for sample in samples:
            extra = getattr(sample, "extra_kwargs", None) or {}
            raw = extra.get(extra_key)
            if raw is None:
                complete = False
                break
            if isinstance(raw, torch.Tensor):
                values.append(float(raw.item()))
            else:
                values.append(float(raw))
        ok = torch.tensor(
            1 if complete and len(values) == len(samples) else 0,
            device=self.accelerator.device,
            dtype=torch.int64,
        )
        ok = self.accelerator.reduce(ok, reduction="sum")
        if int(ok.item()) < int(self.accelerator.num_processes):
            return None
        local = torch.tensor(values, device=self.accelerator.device, dtype=torch.float32)
        return self.accelerator.gather(local).cpu().numpy()

    def _gather_component_score_logs(
        self, samples: List[BaseSample], split: str = "eval"
    ) -> Dict[str, float]:
        """Gather TextPecker / TextDART extras without changing scalar rewards.

        TextPecker: ``semantic_scores`` / ``quality_scores``.
        TextDART: ``region_whole`` (targets + extra 0-1) and ``region_target``
        (targets only). The TextDART scalar is already
        ``{split}/reward_{name}_mean`` from ``RewardModelOutput.rewards``.
        Missing keys skip that component so OCR-only recipes stay unchanged.
        """
        if split not in ("eval", "train"):
            raise ValueError(f"split must be 'eval' or 'train', got {split!r}")
        logs: Dict[str, float] = {}
        key_to_tag = (
            ("semantic_scores", "semantic"),
            ("quality_scores", "quality"),
        )
        for extra_key, tag in key_to_tag:
            gathered = self._gather_per_sample_extra_floats(samples, extra_key)
            if gathered is None:
                continue
            logs[f"{split}/reward_textpecker_{tag}_mean"] = float(np.mean(gathered))
            logs[f"{split}/reward_textpecker_{tag}_std"] = float(np.std(gathered))
        textdart_names = self._textdart_reward_names(split)
        if textdart_names:
            for extra_key in ("region_whole", "region_target"):
                gathered = self._gather_per_sample_extra_floats(samples, extra_key)
                if gathered is None:
                    continue
                for name in textdart_names:
                    logs[f"{split}/reward_{name}_{extra_key}_mean"] = float(np.mean(gathered))
                    logs[f"{split}/reward_{name}_{extra_key}_std"] = float(np.std(gathered))
        return logs

    def _init_logging_backend(self):
        """Initialize logging backend if specified."""
        if self.accelerator.is_main_process:
            self.logger = load_logger(self.config)
        else:
            self.logger = None
        self.accelerator.wait_for_everyone()

    def _init_reward_model(self) -> Tuple[Dict[str, BaseRewardModel], Dict[str, BaseRewardModel]]:
        """Initialize reward model from configuration."""

        # If DeepSpeed ZeRO-3 is enabled, the reward model will be somehow sharded.
        # We need to disable ZeRO-3 init context when loading the model to avoid issues
        # NOTE: This bug persists even with this context manager. DONOT USE ZeRO-3.
        # A possible solution: use DeepSpeed GatherParamter manually in the reward_model's `forward`.

        # Initialize all reward model instances
        self.reward_loader = MultiRewardLoader(
            reward_args=self.config.reward_args,
            accelerator=self.accelerator,
            eval_reward_args=self.config.eval_reward_args,
        ).load()
        # Get training & eval reward models
        self.reward_models = self.reward_loader.get_training_reward_models()
        self.eval_reward_models = self.reward_loader.get_eval_reward_models()
        train_reward_configs = self.reward_loader.get_reward_configs("train")
        eval_reward_configs = self.reward_loader.get_reward_configs("eval")
        # Initialize reward processor
        group_on_same_rank = self.config.data_args.sampler_type == "group_contiguous"
        self.reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.reward_models,
            reward_configs=train_reward_configs,
            tokenizer=self.adapter.tokenizer,  # For prompt encoding/decoding,
            group_on_same_rank=group_on_same_rank,
            verbose=self.log_args.verbose,
        )
        self.eval_reward_processor = RewardProcessor(
            accelerator=self.accelerator,
            reward_models=self.eval_reward_models,
            reward_configs=eval_reward_configs,
            tokenizer=self.adapter.tokenizer,  # For prompt encoding/decoding
            group_on_same_rank=group_on_same_rank,
            verbose=self.log_args.verbose,
        )
        # Initialize reward buffers
        self.reward_buffer = RewardBuffer(
            self.reward_processor,
            self.training_args.group_size,
        )
        self.eval_reward_buffer = RewardBuffer(
            self.eval_reward_processor,
            self.training_args.group_size,
        )

        # Initialize advantage processor
        self.advantage_processor = AdvantageProcessor(
            accelerator=self.accelerator,
            reward_weights={name: cfg.weight for name, cfg in train_reward_configs.items()},
            group_size=self.training_args.group_size,
            global_std=getattr(self.training_args, "global_std", True),
            sampler_type=self.config.data_args.sampler_type,
            verbose=self.log_args.verbose,
        )

        return self.reward_models, self.eval_reward_models

    def _init_dataloader(self) -> Tuple[DataLoader, Union[None, DataLoader]]:
        # Move text-encoder & vae to GPU for dataloader encoding
        self.adapter.on_load_components(
            components=self.adapter.preprocessing_modules, device=self.accelerator.device
        )
        dataloader, test_dataloader = get_dataloader(
            config=self.config,
            accelerator=self.accelerator,
            preprocess_func=self.adapter.preprocess_func,
        )
        # Offload text-encoder after dataloader encoding
        self.adapter.off_load_components(
            components=self.adapter.preprocessing_modules,
        )

        self.accelerator.wait_for_everyone()

        return dataloader, test_dataloader

    def _epoch_data_iter(self) -> Iterator[Dict[str, Any]]:
        """Fresh train-dataloader iterator whose batches belong to ``self.epoch``.

        Every sampler draws its prompt subset from ``seed + sampler.epoch`` and
        advances that counter itself while the worker prefetch pulls indices, so
        the schedule would otherwise depend on the prefetch depth and restart
        from the first epoch's prompts after a resume. Pinning it here makes the
        prompt schedule a pure function of the trainer epoch.
        """
        sampler = self.dataloader.batch_sampler
        if not hasattr(sampler, "set_epoch"):
            raise TypeError(
                "expected the train dataloader's batch_sampler to be a Flow-Factory "
                f"sampler exposing set_epoch(epoch), got {type(sampler).__name__}. "
                "Samplers must be built by `data_utils.sampler_loader.get_data_sampler`."
            )
        sampler.set_epoch(self.epoch)
        return iter(self.dataloader)

    def _init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.adapter.get_trainable_parameters(),
            lr=self.training_args.learning_rate,
            betas=self.training_args.adam_betas,
            weight_decay=self.training_args.adam_weight_decay,
            eps=self.training_args.adam_epsilon,
        )
        return self.optimizer

    def _load_inference_components(self, trainable_module_names: List[str]):
        """
        Load non-trainable components needed at runtime to the accelerator device.

        Trainable modules are already on-device via `accelerator.prepare()`.
        This loads the remaining modules required for inference and,
        when preprocessing is disabled, also loads encoding components
        that would otherwise stay offloaded.
        """
        prepared_names = set(trainable_module_names)

        modules_to_load = list(self.adapter.inference_modules)

        if not self.config.data_args.enable_preprocess:
            modules_to_load.extend(self.adapter.preprocessing_modules)

        # Resolve group names → concrete names, then deduplicate & exclude prepared
        resolved = self.adapter._resolve_component_names(modules_to_load)
        resolved = [m for m in resolved if m not in prepared_names]

        if resolved:
            self.adapter.on_load_components(
                components=resolved,
                device=self.accelerator.device,
            )

    def _initialization(self):
        # Fix for FSDP, synchronize frozen components like text encoder & VAE.
        # Otherwise they may be uninitialized on Rank > 0.
        if self.adapter._is_fsdp_cpu_efficient_loading():
            logger.info("FSDP CPU Efficient Loading detected. Synchronizing frozen components...")
            # self.adapter.on_load(self.accelerator.device)
            self._synchronize_frozen_components()

        # Init dataloader and optimizer
        self.dataloader, self.test_dataloader = self._init_dataloader()

        # Dynamically get all trainable modules from target_module_map
        trainable_module_names = list(self.adapter.target_module_map.keys())

        self.optimizer = self._init_optimizer()
        # Prepare everything with accelerator
        trainable_modules = [
            getattr(self.adapter, name)
            for name in trainable_module_names
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None
        ]
        # Prepare trainable modules + optimizer + test_dataloader
        to_prepare = trainable_modules + [self.optimizer]
        if self.test_dataloader is not None:
            to_prepare.append(self.test_dataloader)

        prepared = self.accelerator.prepare(*to_prepare)
        # Here, `self.dataloader` is not prepared since it has been handled with DistributedKRepeatSampler
        for i, name in enumerate(trainable_module_names):
            if hasattr(self.adapter, name) and getattr(self.adapter, name) is not None:
                self.adapter.set_component(name, prepared[i])

        self.optimizer = prepared[len(trainable_modules)]
        if self.test_dataloader is not None:
            self.test_dataloader = prepared[len(trainable_modules) + 1]

        # Load inference modules, excluding already-prepared ones
        self._load_inference_components(trainable_module_names)

        # Initialize reward model
        self._init_reward_model()

    def _synchronize_frozen_components(self):
        if self.accelerator.num_processes <= 1:
            return

        # Synchronize all non-prepared components
        all_names = self.adapter._resolve_component_names()
        for name in all_names:
            if self.adapter._should_manage_device(name):
                comp = self.adapter.get_component(name)
                if comp is not None:
                    for param in comp.parameters():
                        param.data = param.data.to(self.accelerator.device)
                        dist.broadcast(param.data, src=0)

        # Barrier to ensure everyone is done
        self.accelerator.wait_for_everyone()
        logger.info(f"[Rank {self.accelerator.process_index}] Frozen components synchronized.")

    @staticmethod
    def _patch_deepspeed_autocast(accelerator):
        """Patch DeepSpeed >=0.17.2 to allow external torch.autocast contexts.

        In v0.17.2+, engine.forward() calls validate_nested_autocast() which
        raises AssertionError if torch.autocast is active outside the engine,
        then wraps the forward with torch.autocast(enabled=torch_autocast_enabled).
        When torch_autocast is not configured (the default for bf16 built-in
        mixed-precision), this inner context uses enabled=False, which explicitly
        *disables* any outer autocast and causes dtype mismatches.

        This patch makes the engine transparent to an outer autocast context:
        validate_nested_autocast becomes a no-op, and torch_autocast_enabled /
        torch_autocast_dtype fall through to the active torch.autocast state so
        the engine re-enables (rather than disables) autocast during forward.
        """
        if getattr(accelerator.state, "deepspeed_plugin", None) is None:
            return

        try:
            import deepspeed.runtime.torch_autocast as _ds_ac
            from deepspeed.runtime.engine import DeepSpeedEngine
        except ImportError:
            return

        if getattr(DeepSpeedEngine, "_ff_autocast_patched", False):
            return

        if hasattr(_ds_ac, "validate_nested_autocast"):
            _ds_ac.validate_nested_autocast = lambda engine: None

        if hasattr(DeepSpeedEngine, "torch_autocast_enabled"):
            _orig_enabled = DeepSpeedEngine.torch_autocast_enabled
            _orig_dtype = DeepSpeedEngine.torch_autocast_dtype

            def _patched_enabled(self):
                return _orig_enabled(self) or torch.is_autocast_enabled()

            def _patched_dtype(self):
                if not _orig_enabled(self) and torch.is_autocast_enabled():
                    return torch.get_autocast_gpu_dtype()
                return _orig_dtype(self)

            DeepSpeedEngine.torch_autocast_enabled = _patched_enabled
            DeepSpeedEngine.torch_autocast_dtype = _patched_dtype

        DeepSpeedEngine._ff_autocast_patched = True

    @abstractmethod
    def start(self, *args, **kwargs):
        """Start training process."""
        pass

    @abstractmethod
    def prepare_feedback(self, samples: List[BaseSample]) -> None:
        """Stages 4--5: finalize rewards, compute advantages, and log metrics (no policy gradients).

        Algorithms that need extra batching before the loss (e.g. DPO chosen/rejected pairs) may
        perform that work in :meth:`optimize` after advantages are on each sample.
        """
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        """Update policy model"""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluation for one epoch."""
        pass

    @staticmethod
    def _attach_batch_metadata(batch: Dict[str, Any], samples: List[BaseSample]) -> None:
        """Propagate per-sample ``metadata`` from the dataloader batch onto generated samples.

        :class:`flow_factory.data_utils.dataset.GeneralDataset` packs every
        non-preprocess column from the raw dataset (e.g. ``target``, ``gt``,
        ``unique_id``, ...) into a per-row ``metadata`` dict via
        ``_preprocess_batch``. ``adapter.inference`` does not know or care
        about that field — its signature is filtered by
        :func:`filter_kwargs` and ``metadata`` is dropped. This helper closes
        the gap by writing ``batch['metadata'][i]`` into
        ``samples[i].extra_kwargs['metadata']`` so reward models can consume
        it via the standard reward-processor field-routing path
        (``filter_kwargs(model.__call__, **sample)``).

        Args:
            batch: The dataloader batch dict for the inference call that produced
                ``samples``. ``batch.get('metadata')`` is expected to be either
                ``None`` (no metadata column) or a ``List[Dict[str, Any]]`` whose
                length equals ``len(samples)``.
            samples: Generated samples from ``adapter.inference(**batch)``.

        Raises:
            ValueError: If ``metadata`` is present but its length does not match
                ``len(samples)`` — this would silently misalign target/sample
                pairs and corrupt reward signals.
        """
        if not isinstance(batch, dict):
            return
        metas = batch.get("metadata")
        if metas is None:
            return
        if not isinstance(metas, list):
            raise TypeError(
                f"expected batch['metadata'] to be a List[Dict], " f"got {type(metas).__name__}"
            )
        if len(metas) != len(samples):
            raise ValueError(
                f"metadata length {len(metas)} does not match samples length "
                f"{len(samples)} after adapter.inference()"
            )
        for sample, meta in zip(samples, metas):
            if meta is None:
                continue
            sample.extra_kwargs["metadata"] = meta

    def _maybe_offload_samples_to_cpu(self, samples: List[BaseSample]) -> None:
        """Move every sample's tensor fields to CPU when ``offload_samples_to_cpu`` is enabled.

        Producer-side half of the CPU-offload + lazy-reload pipeline: samples
        leave ``sample()`` already on CPU so that the GPU peak from the rollout
        buffer is bounded by a single batch worth of inference activations.

        Must be called BEFORE ``self.reward_buffer.add_samples(...)`` so that
        the buffer's recorded ``sync_event`` captures "D2H complete + data
        ready on CPU"; downstream reward workers (sync or async) then see a
        deterministic CPU-resident state and trigger their own H2D inside
        ``RewardProcessor`` (see ``move_tensors_to_device`` in
        ``utils/base.py``).

        No-op when ``training_args.offload_samples_to_cpu`` is False
        (default); samples remain on the GPU.

        Args:
            samples: Newly generated samples for the current sample loop iteration.
        """
        if not self.training_args.offload_samples_to_cpu:
            return
        for sample in samples:
            sample.to("cpu")

    _MAX_WANDB_TRAIN_GROUPS = 4
    """Max number of prompt groups to log to wandb per epoch."""

    def _build_train_sample_log_payload(self, samples: List[BaseSample]) -> Any:
        """Build the payload logged under ``train_samples`` (wandb media panel).

        Default: pass the samples through unchanged — the logger renders each
        ``sample.image`` with a reward/prompt caption. Subclasses may override
        to add visual overlays (e.g. drawing per-target bounding boxes). This
        affects ONLY the training-time media panel; disk saving (below) and the
        evaluation logging path are intentionally left untouched.
        """
        return samples

    def _save_sample_images(self, samples: List[BaseSample], epoch: int) -> None:
        """Post-feedback sample processing: log to wandb and optionally save to disk.

        **Wandb logging** (always, on main process with a logger):
        Logs up to ``_MAX_WANDB_TRAIN_GROUPS`` prompt groups as ``train_samples``
        so training-time generation quality is visible in the media panel.

        **Disk saving** (when ``log.save_samples`` is ``true``):

        Layout::

            {save_samples_dir}/{run_name}/{epoch}/{group_id}/sample_0.png ...
            {save_samples_dir}/{run_name}/{epoch}/metadata.jsonl

        Only runs on the main process. Samples are grouped by ``unique_id``
        (same prompt = same group). ``metadata.jsonl`` records one JSON object
        per group with prompt, target, and any other dataset metadata.
        """
        if not self.accelerator.is_main_process:
            return

        # Group samples by unique_id (prompt identity)
        groups: Dict[int, List[BaseSample]] = defaultdict(list)
        for sample in samples:
            groups[sample.unique_id].append(sample)

        # 1. Log a subset of training samples to wandb
        if self.logger is not None:
            log_samples: List[BaseSample] = []
            for idx, group_samples in enumerate(groups.values()):
                if idx >= self._MAX_WANDB_TRAIN_GROUPS:
                    break
                log_samples.extend(group_samples)
            if log_samples:
                train_payload = self._build_train_sample_log_payload(log_samples)
                self.log_data({"train_samples": train_payload}, step=self.step)

        # 2. Save to disk (optional)
        if not self.log_args.save_samples:
            return

        step_dir = os.path.join(
            self.log_args.save_samples_dir,
            str(self.log_args.run_name),
            str(epoch),
        )
        os.makedirs(step_dir, exist_ok=True)

        metadata_records: List[Dict[str, Any]] = []

        for group_idx, (_uid, group_samples) in enumerate(groups.items()):
            group_dir = os.path.join(step_dir, str(group_idx))
            os.makedirs(group_dir, exist_ok=True)

            for sample_idx, sample in enumerate(group_samples):
                if sample.image is not None:
                    pil_images = standardize_image_batch(sample.image, output_type="pil")
                    for img_idx, pil_img in enumerate(pil_images):
                        if len(pil_images) == 1:
                            filename = f"sample_{sample_idx}.png"
                        else:
                            filename = f"sample_{sample_idx}_{img_idx}.png"
                        pil_img.save(os.path.join(group_dir, filename))

            representative = group_samples[0]
            record: Dict[str, Any] = {"group_id": group_idx}
            if representative.prompt is not None:
                record["prompt"] = representative.prompt
            if representative.negative_prompt is not None:
                record["negative_prompt"] = representative.negative_prompt

            meta = representative.extra_kwargs.get("metadata")
            if isinstance(meta, dict):
                for key, value in meta.items():
                    if key not in record:
                        record[key] = value

            record["num_samples"] = len(group_samples)

            rewards_list = []
            for s in group_samples:
                r = s.extra_kwargs.get("rewards")
                if r is not None:
                    rewards_list.append(r)
            if rewards_list:
                record["rewards"] = rewards_list

            metadata_records.append(record)

        metadata_path = os.path.join(step_dir, "metadata.jsonl")
        with open(metadata_path, "w", encoding="utf-8") as f:
            for rec in metadata_records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        logger.info(
            "Saved %d groups (%d samples) to %s",
            len(groups),
            len(samples),
            step_dir,
        )

    _TRAINER_STATE_FILENAME = "trainer_state.json"

    def _optimizer_steps_per_epoch(self) -> int:
        """Number of optimizer (``self.step``) increments performed per epoch.

        The optimize loop runs ``num_inner_epochs * num_batches_per_epoch *
        num_train_timesteps`` micro-batches and steps the optimizer once every
        ``gradient_accumulation_steps`` of them, so::

            steps_per_epoch = (num_inner_epochs * num_batches_per_epoch
                               * num_train_timesteps) // gradient_accumulation_steps

        This holds for the GRPO family (incl. GRPO-Guard / SCANNER),
        which is the only path that supports resume. It is used purely as a
        fallback to align ``self.step`` with ``epoch`` when
        ``trainer_state.json`` is absent.
        """
        ta = self.training_args
        num_train_timesteps = ta.get_num_train_timesteps(self.config)
        micro_steps = ta.num_inner_epochs * ta.num_batches_per_epoch * num_train_timesteps
        return max(1, micro_steps // ta.gradient_accumulation_steps)

    def _restore_training_progress(self) -> None:
        """Restore ``self.epoch`` / ``self.step`` when continuing a run.

        Preference order:

        1. ``trainer_state.json`` inside ``resume_path`` — exact counters saved
           alongside the checkpoint (written by :meth:`save_checkpoint`).
        2. Fallback for checkpoints without that file: parse the epoch
           from the ``checkpoint-{epoch}`` directory name and derive the step
           as ``epoch * _optimizer_steps_per_epoch()``.

        Raises:
            ValueError: If neither source can determine the epoch (e.g. the
                resume path is not a ``checkpoint-*`` directory).
        """
        resume_path = self.model_args.resume_path
        state_path = os.path.join(resume_path, self._TRAINER_STATE_FILENAME)

        if os.path.isfile(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if "epoch" not in state or "step" not in state:
                raise ValueError(
                    f"{state_path} is missing required keys 'epoch'/'step'; "
                    f"got keys {sorted(state.keys())}."
                )
            self.epoch = int(state["epoch"])
            self.step = int(state["step"])
            logger.info(
                "Resuming run from %s: epoch=%d, step=%d (from trainer_state.json).",
                resume_path,
                self.epoch,
                self.step,
            )
            return

        parsed = self.config.parse_checkpoint_path(resume_path)
        if parsed is None:
            raise ValueError(
                "log.resume_run=True but could not restore progress: "
                f"{resume_path!r} has no '{self._TRAINER_STATE_FILENAME}' and is "
                "not a 'checkpoint-{epoch}' directory. Point resume_path at a "
                "Flow-Factory checkpoint or set log.resume_run=False."
            )
        _save_dir, _run_name, epoch = parsed
        self.epoch = epoch
        self.step = epoch * self._optimizer_steps_per_epoch()
        logger.info(
            "Resuming run from %s: epoch=%d, step=%d "
            "(derived from checkpoint name; no trainer_state.json found).",
            resume_path,
            self.epoch,
            self.step,
        )

    def save_checkpoint(self, save_directory: str, epoch: Optional[int] = None):
        """Save trainer state to a specific path.

        All ranks must call this method (collective state-dict gathering may
        be required for FSDP / DeepSpeed ZeRO-3).  The adapter internally
        ensures only the main process writes files to disk.
        """
        if epoch is not None:
            save_directory = os.path.join(save_directory, f"checkpoint-{epoch}")

        self.adapter.save_checkpoint(
            save_directory=save_directory,
            model_only=self.log_args.save_model_only,
        )

        # Persist epoch/step next to the weights so a later `resume_run` lands
        # on the exact counters (x-axis stays aligned across restarts).
        if self.accelerator.is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            state_path = os.path.join(save_directory, self._TRAINER_STATE_FILENAME)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": int(epoch) if epoch is not None else int(self.epoch),
                        "step": int(self.step),
                        "run_name": str(self.log_args.run_name),
                    },
                    f,
                    indent=2,
                )

    def load_checkpoint(
        self,
        path: str,
        resume_type: Optional[Literal["lora", "full", "state"]] = None,
    ):
        """Load trainer state from a specific path."""
        self.adapter.load_checkpoint(
            path=path,
            strict=True,
            resume_type=resume_type,
        )
        self.accelerator.wait_for_everyone()

    def cleanup(self) -> None:
        """Shutdown reward workers and flush the logging backend.

        Called on KeyboardInterrupt to cancel pending futures and signal
        executor threads to stop. Also calls ``logger.finish()`` so that
        platforms like wandb can sync buffered data before ``os._exit()``
        kills the process (``os._exit`` bypasses ``atexit`` handlers,
        so without this call all un-synced wandb metrics are lost).
        """
        for buf in (
            getattr(self, "reward_buffer", None),
            getattr(self, "eval_reward_buffer", None),
        ):
            if buf is not None:
                buf.shutdown(wait=False, cancel_futures=True)
        if getattr(self, "logger", None) is not None:
            self.logger.finish()
