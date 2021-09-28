import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
import torch.distributed as dist
import math

def cuda_variable(tensor, non_blocking=False):
    if torch.cuda.is_available():
        # return tensor.to('cuda', non_blocking=non_blocking)
        return tensor.to(f'cuda:{dist.get_rank()}', non_blocking=non_blocking)
    else:
        return tensor


def is_main_process():
    return dist.get_rank() == 0


def get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]
    

def all_reduce_scalar(scalar, average=True):
    t = torch.Tensor([scalar]).to(f'cuda:{dist.get_rank()}')
    dist.all_reduce(t)
    scalar = t[0].detach().item()
    if average:
        scalar = scalar / dist.get_world_size()
    return scalar


def display_monitored_quantities(epoch_id, monitored_quantities_train,
                                 monitored_quantities_val) -> None:
    if is_main_process():
        print(f'======= Epoch {epoch_id} =======')
        print(f'---Train---')
        dict_pretty_print(monitored_quantities_train, endstr=' ' * 5)
        print()
        print(f'---Val---')
        dict_pretty_print(monitored_quantities_val, endstr=' ' * 5)
        print('\n')


def to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()


def dict_pretty_print(d, endstr='\n'):
    for key, value in d.items():
        if type(value) == list:
            print(f'{key.capitalize()}: [%s]' % ', '.join(map(str, value)))
        else:
            print(f'{key.capitalize()}: {value:.6}', end=endstr)


def chorale_accuracy(value, target):
    """
    :param value: list of (batch_size, chorale_length, num_notes)
    :param target: (batch_size, num_voices, chorale_length)
    :return:
    """
    batch_size, num_voices, chorale_length = target.size()
    batch_size, chorale_length, _ = value[0].size()
    num_voices = len(value)

    # put num_voices first
    target = target.transpose(0, 1)

    sum = 0
    for voice, voice_target in zip(value, target):
        max_values, max_indexes = torch.max(voice, dim=2, keepdim=False)
        num_correct = (max_indexes == voice_target).float().mean().item()
        sum += num_correct

    return sum / num_voices


def log_normal(x, mean, log_var, eps=0.00001):
    return -(x - mean)**2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + c


def categorical_crossentropy(value, target, mask=None, label_smoothing=False):
    """

    :param value: list of (batch_size, num_events, num_tokens_of_corresponding_channel)
    :param target: (batch_size, num_events, num_channels)
    :param mask: (batch_size, num_events, num_channels)
    :return:
    """
    cross_entropy = nn.CrossEntropyLoss(size_average=False, reduce=False)
    sum = 0

    for channel_probs, target_channel, mask_channel in zip(
            value, target.split(1, dim=2), mask.split(1, dim=2)):
        # select relevent indices
        batch_size, num_events, num_tokens_of_channel = channel_probs.size()

        probs = channel_probs[mask_channel.bool().repeat(
            1, 1, num_tokens_of_channel)]
        target = target_channel[mask_channel.bool()]

        probs = probs.view(-1, num_tokens_of_channel)
        tgt = target.view(-1)

        if not label_smoothing:
            ce = cross_entropy(probs, tgt)
        else:
            eps = 0.02
            one_hot = torch.zeros_like(probs).scatter(1, tgt.view(-1, 1), 1)
            one_hot = (one_hot * (1 - eps) +
                       (1 - one_hot) * eps / (num_tokens_of_channel - 1)
                       )
            log_prb = nn.functional.log_softmax(probs, dim=1)
            ce = -(one_hot * log_prb).sum(dim=1)
        sum = sum + ce.sum()

    # divide by the total number of tokens
    if mask.sum() > 0:
        sum = sum / mask.sum()
    return sum


def distilled_categorical_crossentropy(value, target, mask=None):
    """

    :param value: list of (batch_size, num_events, num_notes)
    :param target: list of (batch_size, num_events, num_notes)
    :return:
    """
    def cross_entropy_from_logits(p, q):
        """
        sum softmax(p) log softmax(q)
        :param p:
        :param q:
        :return:
        """
        p = torch.softmax(p, dim=1)
        log_term = q - torch.logsumexp(q, dim=1, keepdim=True)
        return -torch.sum(p * log_term, dim=1)

    def cross_entropy_from_logits_check(p, q):
        """
        sum softmax(p) log softmax(q)
        :param p:
        :param q:
        :return:
        """
        p = torch.softmax(p, dim=1)
        q = torch.softmax(q, dim=1)
        return -torch.sum(p * torch.log(q), dim=1)

    sum = 0
    # TODO better mask horrible
    for channel, channel_target, channel_mask in zip(value, target,
                                                     mask.split(1, dim=2)):
        # channel is (batch_size, num_events, num_tokens_of_corresponding_channel)
        # channel_target is (batch_size, num_events)
        # TODO remove this loop
        for probs, label, m in zip(channel.split(1, dim=1),
                                   channel_target.split(1, dim=1),
                                   channel_mask.split(1, dim=1)):
            if m.squeeze(2).squeeze(1).float().mean().item() > 0.5:
                ce = cross_entropy_from_logits(label.squeeze(1),
                                               probs.squeeze(1))
                sum = sum + ce
    return sum


def quantization_loss(loss_quantization_left, loss_quantization_negative,
                      loss_quantization_right):
    loss_quantization = torch.cat((
        loss_quantization_left.sum(2).sum(1),
        loss_quantization_right.sum(2).sum(1),
        loss_quantization_negative.sum(4).sum(3).sum(2).sum(1),
    ),
        dim=0).mean()
    return loss_quantization


def quantization_loss_no_negative(loss_quantization_left,
                                  loss_quantization_right):
    loss_quantization = torch.cat((
        loss_quantization_left.sum(2).sum(1),
        loss_quantization_right.sum(2).sum(1),
    ),
        dim=0).mean()
    return loss_quantization


def to_sphere(t):
    return t / t.norm(dim=-1, keepdim=True)


def flatten(x):
    """

    :param x:(batch, num_events, num_channels, ...)
    :return: (batch, num_events * num_channels, ...) with num_channels varying faster
    """
    size = x.size()
    assert len(size) >= 3
    batch_size, num_events, num_channels = size[:3]
    remaining_dims = list(size[3:])
    x = x.view(batch_size, num_events * num_channels, *remaining_dims)
    return x


def unflatten(sequence, num_channels):
    """

    :param sequence: (batch_size, num_events * num_channels, ...)
    where num_channels is varying faster
    :return: (batch_size, num_events, num_channels, ...)
    """
    size = sequence.size()
    assert len(size) >= 2
    batch_size, sequence_length = size[:2]
    assert sequence_length % num_channels == 0
    num_events = sequence_length // num_channels
    remaining_dims = list(size[2:])

    chorale = sequence.view(batch_size, num_events, num_channels,
                            *remaining_dims)
    return chorale


def timing_gpu():
    """
    Just to remember how to time gpus operation
    :return:
    """

    # notes ##################################
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'T: {elapsed_time_ms}')
    # notes ##################################


def plot_mi_marginals(px, py, mi_matrix, save_path):
    nullfmt = NullFormatter()  # no labels
    dim_x = len(px)
    dim_y = len(py)
    dim_max = max(dim_x, dim_y)

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    # plt.figure(1, figsize=(dim_max, dim_max))
    # Or fixed size perhaps ??
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Plot MI
    im = axScatter.imshow(mi_matrix, cmap='RdBu')
    divider = make_axes_locatable(axScatter)
    # create an axes on the right side of ax. The width of cax will be 1%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    cax = divider.append_axes("left", size="5%", pad=0.05)
    axScatter.figure.colorbar(im, cax=cax)
    axScatter.set_xlim((-0.5, dim_max - 0.5))
    axScatter.set_ylim((-0.5, dim_max - 0.5))

    # Plot marginals
    colorbar_width = dim_max * 0.05
    axHistx.bar(x=range(dim_y), height=py)
    axHisty.barh(y=range(dim_x), width=np.flip(px))
    axHistx.set_xlim((-0.5 - colorbar_width, dim_max - 0.5))
    axHisty.set_ylim((-0.5, dim_max - 0.5))

    plt.savefig(save_path)
    return


def top_k_top_p_filtering(logits,
                          top_k=0,
                          top_p=0.0,
                          filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    # batch size 1 for now - could be updated for more but the code would be less clear
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    # Pytorch ordering
    axis = len(x.size()) - 3
    return torch.nn.ELU()(torch.cat([x, -x], dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(
        torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(
        xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + cuda_variable(torch.zeros(xs + [nr_mix]))
    m2 = (means[:, :, :, 1, :] +
          coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(
              xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(
              xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - torch.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -torch.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * torch.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(
        torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
            log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (
        1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(torch.logsumexp(log_probs))


def discretized_mix_logistic_loss_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :,
          nr_mix:].contiguous().view(xs + [nr_mix * 2])  # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + cuda_variable(torch.zeros(xs + [nr_mix]).cuda())

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - torch.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -torch.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * torch.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(
        torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
            log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (
        1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(torch.logsumexp(log_probs))

class Adafactor(torch.optim.Optimizer):
    """Implements Adafactor algorithm.
    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)
    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constans for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        beta1 (float): coefficient used for computing running averages of gradient
            (default: None)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super(Adafactor, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = (
                1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                group["lr"] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad ** 2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2t
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2t
                    )

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )
                update.mul_(group["lr"])

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss