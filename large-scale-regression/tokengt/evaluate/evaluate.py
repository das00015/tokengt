"""
Modified from https://github.com/microsoft/Graphormer
"""

import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score


import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pretrain import load_pretrained_model

import logging


def eval(args, use_pretrained, checkpoint_path=None, logger=None):
    #print("entered eval ")
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    #print("cfg.task")
    # initialize task
    task = tasks.setup_task(cfg.task)
    #print(task)
    model = task.build_model(cfg.model) 
    #print(model)
    #print("model is loaded")
    # load checkpoint
    if use_pretrained:
        model_state = load_pretrained_model(cfg.task.pretrained_model_name)
    else:
        model_state = torch.load(checkpoint_path)["model"]
        print("checkpoint loaded")
        model_state.pop('encoder.graph_encoder.graph_feature.orf_encoder.weight', None)
        #print(model_state)
    
    model.load_state_dict(
        model_state, strict=True, model_cfg=cfg.model # error is here!!
    )
    #print("model_state_loaded")
    del model_state
    #print("model to device")
    model.to(torch.cuda.current_device())
    # load dataset
    split = args.split
    print("split downloaded", split)
    task.load_dataset(split)
    print("data loaded")
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )
    print("eval starts")
    # infer
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample)
            y = model(**sample["net_input"])
            y = y.reshape(-1)
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()

    # save predictions
    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)

    # evaluate pretrained models
    if use_pretrained:
        if cfg.task.pretrained_model_name == "pcqm4mv1_sgt_base":
            evaluator = ogb.lsc.PCQM4MEvaluator()
            input_dict = {'y_pred': y_pred, 'y_true': y_true}
            result_dict = evaluator.eval(input_dict)
            logger.info(f'PCQM4Mv1Evaluator: {result_dict}')
        elif cfg.task.pretrained_model_name == "pcqm4mv2_sgt_base":
            evaluator = ogb.lsc.PCQM4Mv2Evaluator()
            input_dict = {'y_pred': y_pred, 'y_true': y_true}
            result_dict = evaluator.eval(input_dict)
            logger.info(f'PCQM4Mv2Evaluator: {result_dict}')
    else:
        if args.metric == "auc":
            auc = roc_auc_score(y_true, y_pred)
            logger.info(f"auc: {auc}")
            # Add these lines for additional metrics
            threshold = 0.5
            pred_labels = (y_pred >= threshold).int()
            true_labels = y_true.int()

            precision = precision_score(true_labels, pred_labels)
            recall = recall_score(true_labels, pred_labels)
            accuracy = accuracy_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels)

            logger.info(f"precision: {precision}")
            logger.info(f"recall: {recall}")
            logger.info(f"accuracy: {accuracy}")
            logger.info(f"f1 score: {f1}")

        elif args.metric == "mae":
            mae = (y_true - y_pred).abs().mean().item()
            logger.info(f"mae: {mae}")
        else:
            raise ValueError(f"Unsupported metric {args.metric}")


def main():
    parser = options.get_training_parser()
    parser.add_argument(
        "--split",
        type=str,
    )
    parser.add_argument(
        "--metric",
        type=str,
    )
    args = options.parse_args_and_arch(parser, modify_parser=None)
    print("Argument values:")
    print(f"--user-dir: {args.user_dir}")
    print(f"--num-workers: {args.num_workers}")
# Add more print statements for other arguments as needed

    logger = logging.getLogger(__name__)
    if args.pretrained_model_name != "none":
        eval(args, False, logger=logger)
    elif hasattr(args, "save_dir"):
        for checkpoint_fname in os.listdir(args.save_dir):
            checkpoint_path = Path(args.save_dir) / checkpoint_fname
            logger.info(f"evaluating checkpoint file {checkpoint_path}")
            eval(args, False, checkpoint_path, logger)


if __name__ == '__main__':
    main()
