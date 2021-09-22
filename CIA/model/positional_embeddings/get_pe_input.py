import torch


def get_pe_input(dataloader_generator, x_embed, h, metadata_dict, pe_input_type, event_representation):
    # TODO take into account if channels are exapnded or not
    if pe_input_type == 'index':
        length = x_embed.size(1)
        batch_size = x_embed.size(0)
        indices = torch.linspace(
            0, length-1, length, device=x_embed.device)
        pe_input = indices[None, :].repeat(batch_size, 1)
    elif pe_input_type == 'elapsed':
        # if h is None:
        #     h = torch.zeros((x_embed.size(0),)).to(x_embed.device)
        # Original sequence is in prefix order!
        x = metadata_dict['original_sequence']
        _, _, num_channels = x.size()
        elapsed_time = dataloader_generator.get_elapsed_time(x)
        # h = elapsed_time[:, -1]
        # h = h - elapsed_time[:, metadata_dict['decoding_start'] - 1]
        # add zeros
        elapsed_time = torch.cat(
            [
                torch.zeros_like(elapsed_time)[:, :1],
                elapsed_time[:, :-1]
            ],
            dim=1
        )
        if elapsed_time.size(1) > metadata_dict['decoding_start']:
            # we need to have an offset for the generated inpainted region
            elapsed_time[:, metadata_dict['decoding_start']:] = (
                elapsed_time[:, metadata_dict['decoding_start']:] -
                elapsed_time[:, metadata_dict['decoding_start']].unsqueeze(1) +
                elapsed_time[:, 255].unsqueeze(1)
            )
        # TODO scale?! only 10?!
        # elapsed_time = elapsed_time * 100
        # h = h * 100
        if torch.any(elapsed_time < 0):
            print('stop')

        if event_representation:
            elapsed_time_channelized = elapsed_time
        else:
            elapsed_time_channelized = elapsed_time.repeat_interleave(num_channels, dim=1)
        pe_input = elapsed_time_channelized
    return pe_input
