from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
from fairseq.criterions import register_criterion
import logging
import torch
import math
from fairseq import metrics, utils
import torch.nn.functional as F


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss



@register_criterion('auxiliarycriterion')
class AuxiliaryCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            mask_loss_weight,
    ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.mask_loss_weight = mask_loss_weight

    def add_args(parser):
        parser.add_argument('--mask-loss-weight', default=0., type=float,
                            help='weight of mask loss')
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def compute_kl_loss(self, model, net_output, pad_mask=None, reduce=True):
        net_prob = model.get_normalized_probs(net_output, log_probs=True)
        net_prob_tec = model.get_normalized_probs(net_output, log_probs=False)

        p, q = torch.split(net_prob, net_prob.size(0) // 2, dim=0)
        p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0) // 2, dim=0)

        p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
        q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')

        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss


    def forward(self, model, sample, reg_alpha, reduce=True, show=False):

        sample_input = sample['net_input']
        sample_concat_input = {
            'src_tokens': torch.cat([sample_input['src_tokens'], sample_input['src_tokens'].clone()], 0),
            'src_lengths': torch.cat([sample_input['src_lengths'], sample_input['src_lengths'].clone()], 0),
            'prev_output_tokens': torch.cat(
                [sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
        }

        net_output, net_output_auxiliary = model(**sample_concat_input)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        lprobs_auxiliary = model.get_normalized_probs(net_output_auxiliary, log_probs=True)
        lprobs_auxiliary = lprobs_auxiliary.view(-1, lprobs_auxiliary.size(-1))
        target = model.get_targets(sample, net_output)
        target_auxiliary = model.get_targets(sample, net_output_auxiliary)
        pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
        target = torch.cat([target, target.clone()], dim=0)
        target_auxiliary = torch.cat([target_auxiliary, target_auxiliary.clone()], dim=0)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target.view(-1, 1), self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        auxiliary_loss, _ = label_smoothed_nll_loss(
            lprobs_auxiliary, target_auxiliary.view(-1, 1), self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        kl_loss = self.compute_kl_loss(model, net_output, pad_mask)
        loss += reg_alpha * kl_loss
        auxiliary_kl_loss = self.compute_kl_loss(model, net_output_auxiliary, pad_mask)
        auxiliary_loss += reg_alpha * auxiliary_kl_loss

        ntokens = sample['ntokens']
        nsentences = sample['target'].size(0)
        sample_size = sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            "auxiliary_loss": auxiliary_loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, auxiliary_loss, sample_size, logging_output

    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        auxiliary_loss_sum = sum(log.get('auxiliary_loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar('auxiliary_loss', auxiliary_loss_sum / sample_size / math.log(2), sample_size, round=6)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
