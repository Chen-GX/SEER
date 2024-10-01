# ---------- transformer training ----------

import math
import torch.nn as nn
from torch.optim import AdamW # Transformers: This implementation of AdamW is deprecated
from transformers.optimization import get_scheduler, Adafactor
from transformers.trainer_pt_utils import get_parameter_names


def create_optimizer(model, args):
    # decay if not LayerNorm or bias``
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    if args.adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "eps":1e-6,
        }
    optimizer_kwargs["lr"] = args.lr
    
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer

def create_scheduler(optimizer, args):
    warmup_steps = math.ceil(args.num_training_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.num_training_steps,
    )
    return lr_scheduler
