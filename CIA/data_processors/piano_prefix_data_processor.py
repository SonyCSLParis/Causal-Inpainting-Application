from DatasetManager.piano.piano_helper import find_nearest_value
from DatasetManager.piano.piano_midi_dataset import END_SYMBOL, PAD_SYMBOL, START_SYMBOL
from .data_processor import DataProcessor
import torch
import random
from torch import nn
from CIA.utils import cuda_variable


class PianoPrefixDataProcessor(DataProcessor):
    def __init__(self, dataloader_generator, embedding_size, num_events,
                 num_tokens_per_channel, num_events_context):
        super(PianoPrefixDataProcessor,
              self).__init__(embedding_size=embedding_size,
                             num_events=num_events,
                             num_tokens_per_channel=num_tokens_per_channel,
                             add_mask_token=True,
                             num_additional_tokens=2)
        # We need full access to the dataset and dataloader_generator
        self.dataloader_generator = dataloader_generator
        self.num_events_before = num_events_before
        self.num_events_after = num_events_after

        self.placeholder_symbols = nn.Parameter(
            torch.LongTensor(num_tokens_per_channel), requires_grad=False)

        # Start Of Decoding
        self.sod_symbols = nn.Parameter(torch.LongTensor(
            [nt + 1 for nt in num_tokens_per_channel]),
            requires_grad=False)

        self.end_tokens = nn.Parameter(torch.LongTensor([
            self.dataloader_generator.dataset.value2index[feature][END_SYMBOL]
            for feature in self.dataloader_generator.features
        ]),
            requires_grad=False)
        self.pad_tokens = nn.Parameter(torch.LongTensor([
            self.dataloader_generator.dataset.value2index[feature][PAD_SYMBOL]
            for feature in self.dataloader_generator.features
        ]),
            requires_grad=False)

        self.start_tokens = nn.Parameter(torch.LongTensor([
            self.dataloader_generator.dataset.value2index[feature]
            [START_SYMBOL] for feature in self.dataloader_generator.features
        ]),
            requires_grad=False)

    def preprocess(self, x, num_events_inpainted):
        """[summary]

        Args:
            x ([type]):
            decomposes as:

            ======= ======= ======
            before  middle  after

        Returns:
        Sequences of the form:

        ======= ============ ====== ==== ======= ==========
        before  placeholder  after  SOD  middle  END XX XX

        """
        num_events_middle = num_events_inpainted
        sequences_size = self.dataloader_generator.sequences_size
        batch_size, num_events, _ = x.size()
        assert num_events == sequences_size
        assert sequences_size > self.num_events_before + self.num_events_after

        x = cuda_variable(x.long())

        max_num_events_middle = sequences_size - \
            self.num_events_before - self.num_events_after - 3 - 1
        if num_events_middle is None:
            num_events_middle = random.randint(1, max_num_events_middle)
        else:
            assert (num_events_middle > 1) and (num_events_middle < max_num_events_middle), \
                'wrong number of events to be inpainted'

        # we will be adding these at the end of the sequence
        # the 3 accounts for the placeholder, the SOD and END tokens
        remainder_num_events = sequences_size - (self.num_events_before +
                                                 num_events_middle +
                                                 self.num_events_after) - 3

        # Slice x
        x = x[:, :self.num_events_before +
              num_events_middle + self.num_events_after]
        batch_size, num_events, _ = x.size()

        # === Find end tokens in x
        is_end_token = (
            x[:, :, 0] == self.end_tokens[0].unsqueeze(0).unsqueeze(0).repeat(
                batch_size, num_events))

        is_pad_token = (
            x[:, :, 0] == self.pad_tokens[0].unsqueeze(0).unsqueeze(0).repeat(
                batch_size, num_events))

        is_start_token = (
            x[:, :,
              0] == self.start_tokens[0].unsqueeze(0).unsqueeze(0).repeat(
                  batch_size, num_events))

        contains_end_token = (is_end_token.sum(1) >= 1)
        contains_start_token = (is_start_token.sum(1) >= 1)

        # only one of those per sequence
        # Only valid when containes_end_token!!
        end_token_location = torch.argmax(is_end_token.long(), dim=1)
        start_token_location = torch.argmax(is_start_token.long(), dim=1)

        assert remainder_num_events >= 0
        before = x[:, :self.num_events_before]
        middle = x[:, self.num_events_before:self.num_events_before +
                   num_events_middle]
        after = x[:, self.num_events_before +
                  num_events_middle:self.num_events_before +
                  num_events_middle + self.num_events_after]

        placeholder_duration = self.dataloader_generator.get_elapsed_time(middle)[
            :, -1]

        # === Compute Placeholder
        placeholder, placeholder_duration_token = self.compute_placeholder(placeholder_duration=placeholder_duration,
                                                                           batch_size=batch_size)

        new_before_list, new_middle_list, new_after_list = [], [], []

        # TODO batch this?!
        for (b, m, a, p_duration, p_duration_token, c_start_token, c_end_token,
             start_token_l,
             end_token_l) in zip(before, middle, after, placeholder_duration,
                                 placeholder_duration_token,
                                 contains_start_token, contains_end_token,
                                 start_token_location, end_token_location):

            # NOTE: start_token_location is only valid is contains_start_token

            # == if middle contains the start token
            if c_start_token and (self.num_events_before <= start_token_l <
                                  self.num_events_before + num_events_middle):

                # == if middle also contains the end token
                if c_end_token and (
                        self.num_events_before <= end_token_l <
                        self.num_events_before + num_events_middle):
                    # we do not need to append an END token
                    pad_or_end_tokens = self.pad_tokens
                else:
                    # otherwise we do
                    pad_or_end_tokens = self.end_tokens

                # Removes all leading PAD
                # adds an END symbol if necessary and eventually pad
                new_middle = torch.cat(
                    [
                        m[start_token_l -
                          self.num_events_before:],  # removes pad
                        pad_or_end_tokens.unsqueeze(0),  # + 1
                        self.pad_tokens.unsqueeze(0).repeat(
                            start_token_l - self.num_events_before + 1,
                            1),  # number_of_removed events + 1
                    ],
                    dim=0)

                # replace start symbol with SOD symbol
                # note that new_middle has two events more compared to m + remainder_events
                new_middle[0, :] = self.sod_symbols

                # a start symbol must be added to "before"
                new_before = b
                new_before[-1] = self.start_tokens

                # after is untouched
                new_after = a

            # == if START token is in "after"
            elif c_start_token and (self.num_events_before + num_events_middle
                                    <= start_token_l):
                # new middle is only an SOD symbol followed by an END symbol
                new_middle = torch.cat([
                    self.sod_symbols.unsqueeze(0),
                    self.end_tokens.unsqueeze(0),
                    self.pad_tokens.unsqueeze(0).repeat(m.size(0), 1)
                ],
                    dim=0)

                # Trim pad symbols, and PAD
                # no END!
                new_after = torch.cat(
                    [
                        a[start_token_l -
                          (self.num_events_before +
                           num_events_middle):],  # trim leading PAD
                        self.pad_tokens.unsqueeze(0).repeat(
                            (start_token_l -
                             (self.num_events_before + num_events_middle), 1
                             )  # put trimmed PAD ad the end of the sequence
                        )
                    ],
                    dim=0)

                # before is untouched ==
                new_before = b

            # == if END token is in "before"
            elif c_end_token and (end_token_l < self.num_events_before):
                # new middle is only an SOD symbol followed by an END symbol
                new_middle = torch.cat([
                    self.sod_symbols.unsqueeze(0),
                    self.end_tokens.unsqueeze(0),
                    self.pad_tokens.unsqueeze(0).repeat(m.size(0), 1)
                ],
                    dim=0)
                # after is untouched
                new_after = a
                # before is untouched
                new_before = b
            # == if in none of the cases above
            else:
                # only add sod and end to middle (if necessary)

                # == if middle contains the end token
                if c_end_token and (
                        self.num_events_before <= end_token_l <
                        self.num_events_before + num_events_middle):
                    # we do not need to append an END token
                    pad_or_end_tokens = self.pad_tokens
                else:
                    # otherwise we do
                    pad_or_end_tokens = self.end_tokens

                new_middle = torch.cat([
                    self.sod_symbols.unsqueeze(0),
                    m,
                    pad_or_end_tokens.unsqueeze(0),
                ],
                    dim=0)
                new_after = a
                new_before = b

            new_after_list.append(new_after)
            new_before_list.append(new_before)
            new_middle_list.append(new_middle)

        new_middle = torch.stack(new_middle_list, dim=0)
        new_after = torch.stack(new_after_list, dim=0)
        new_before = torch.stack(new_before_list, dim=0)

        # after this new_middle contains 2 additional tokens compared to m
        # starts with SOD and contains only one END symbol

        # creates final sequence
        y = torch.cat([
            new_before, placeholder, new_after, new_middle,
            self.pad_tokens.unsqueeze(0).unsqueeze(0).repeat(
                batch_size, remainder_num_events, 1)
        ],
            dim=1)

        _, num_events_output, _ = y.size()
        # recompute padding mask
        padding_mask = (
            y[:, :, :] == self.pad_tokens.unsqueeze(0).unsqueeze(0).repeat(
                batch_size, num_events_output, 1))
        sod_mask = (
            y[:, :, :] == self.sod_symbols.unsqueeze(0).unsqueeze(0).repeat(
                batch_size, num_events_output, 1))

        start_mask = (
            y[:, :, :] == self.start_tokens.unsqueeze(0).unsqueeze(0).repeat(
                batch_size, num_events_output, 1))

        final_mask = padding_mask + sod_mask + start_mask
        # add placeholder: it is added at num_events_before position
        final_mask[:, self.num_events_before, :] = True

        # decoding_start = self.num_events_before + self.num_events_after + 2
        # # compute decoding_end
        # is_end_token_new_middle = (
        #     new_middle[:, :, 0] == self.end_tokens[0].unsqueeze(0).unsqueeze(0).repeat(
        #         batch_size, new_middle.size(1)))
        # # only valid when containes_end_token!!
        # end_token_location_new_middle = torch.argmax(
        #     is_end_token_new_middle.long(), dim=1)
        # decoding_end = self.num_events_before + \
        #     self.num_events_after + 2 + end_token_location_new_middle

        # self.num_events_before + self.num_events_after + 1 is the location
        # of the SOD symbol (only the placeholder is added)
        metadata_dict = {
            'placeholder_duration': placeholder_duration,
            # 'decoding_start': decoding_start,
            # 'decoding_end': decoding_end,
            'original_sequence': y,
            'loss_mask': final_mask
        }
        return y, metadata_dict

    def compute_placeholder(self, placeholder_duration, batch_size):
        placeholder_duration_token = cuda_variable(
            torch.Tensor([
                self.dataloader_generator.dataset.value2index['time_shift']
                [find_nearest_value(
                    self.dataloader_generator.dataset.time_table_time_shift,
                    pd.item())] for pd in placeholder_duration
            ]))

        # placeholder is batch_size, 1, 4
        placeholder = self.placeholder_symbols.unsqueeze(0).unsqueeze(
            0).repeat(batch_size, 1, 1)
        placeholder[:, 0,
                    self.dataloader_generator.get_feature_index(
                        'time_shift')] = placeholder_duration_token
        return placeholder, placeholder_duration_token

    def compute_elapsed_time(self, metadata_dict):
        # if h is None:
        #     h = torch.zeros((x_embed.size(0),)).to(x_embed.device)
        # Original sequence is in prefix order!
        x = metadata_dict['original_sequence']
        _, _, num_channels = x.size()
        elapsed_time = self.dataloader_generator.get_elapsed_time(x)
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
        return elapsed_time

    def postprocess(self, x, decoding_end, metadata_dict):
        decoding_start = metadata_dict['decoding_start']
        # put all pieces in order:
        x_out = torch.cat(
            [
                x[:, :self.num_events_before],
                x[:, decoding_start: decoding_end],
                x[:, self.num_events_before +
                    1: self.num_events_before + 1 + self.num_events_after]
            ], dim=1
        )
        return x_out
