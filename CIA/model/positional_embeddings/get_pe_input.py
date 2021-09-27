from CIA.data_processors import data_processor
import torch


def get_pe_input(data_processor, x_embed, h, metadata_dict, pe_input_type, event_representation):
    # TODO take into account if channels are exapnded or not
    if pe_input_type == 'index':
        length = x_embed.size(1)
        batch_size = x_embed.size(0)
        indices = torch.linspace(
            0, length-1, length, device=x_embed.device)
        pe_input = indices[None, :].repeat(batch_size, 1)
    elif pe_input_type == 'elapsed':
        elapsed_time = data_processor.compute_elapsed_time(metadata_dict)
        if event_representation:
            elapsed_time_channelized = elapsed_time
        else:
            num_channels = metadata_dict['original_sequence'].shape[-1]
            elapsed_time_channelized = elapsed_time.repeat_interleave(num_channels, dim=1)
        pe_input = elapsed_time_channelized
    return pe_input
