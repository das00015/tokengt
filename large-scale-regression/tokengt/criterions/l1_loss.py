"""
Modified from https://github.com/microsoft/Graphormer
"""

from fairseq.dataclass.configs import FairseqDataclass

import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("l1_loss", dataclass=FairseqDataclass)
class GraphPredictionL1Loss(FairseqCriterion):
    """
    Implementation for the L1 loss (MAE loss) used in tokengt model training --> changed this to BCE!
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"] # total no of samples in the batch.

        with torch.no_grad():
            natoms = max(sample["net_input"]["batched_data"]["node_num"])
    
        # model's forward method unpacks the input data in the dictionary and assigns output of forward to logits
        logits = model(**sample["net_input"])

        # for each sample in the batch; get the logits for the sample. get targets returns the target for each sample.
        targets = model.get_targets(sample, [logits])
        #new_targets = sample["net_input"]["batched_data"]["y"].view(-1,1).float() # we get it directly from the sample. 
        
        # this is done to match the size of the logits tensor along the batch dimension.
        new_targets = targets[: logits.size(0)].float()
        assert (sample["net_input"]["batched_data"]["y"].view(-1,1).float() == new_targets).all()
        assert (max(sample['net_input']['batched_data']['node_num'])<512)
        loss = nn.BCEWithLogitsLoss(reduction="sum")(logits, new_targets) 
        #loss = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)]) # change this to cross entropy

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        #print("Loss calculated for an actual sample of",logging_output['sample_size']," :", loss)
        return loss, sample_size, logging_output # need to explore why sample_size different from logging_output[sample_size]

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        #print("loss sum --> ", loss_sum)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        #print("sample size -->", sample_size)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
        #metrics.log_scalar("loss", loss_sum)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("l1_loss_with_flag", dataclass=FairseqDataclass)
class GraphPredictionL1LossWithFlag(FairseqCriterion):
    """
    Implementation for the binary log loss used in tokengt model training.
    """

    def forward(self, model, sample, perturb=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = max(sample["net_input"]["batched_data"]["node_num"])

        logits = model(**sample["net_input"], perturb=perturb)
        targets = model.get_targets(sample, [logits])

        loss = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)])

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
