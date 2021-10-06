from DatasetManager.piano.piano_helper import find_nearest_value
from DatasetManager.piano.piano_midi_dataset import END_SYMBOL, PAD_SYMBOL, START_SYMBOL
from .data_processor import DataProcessor
import torch
import random
from torch import nn
from CIA.utils import cuda_variable


class PianoPrefixEndDataProcessor(DataProcessor):
    def __init__(self, dataloader_generator, embedding_size, num_events,
                 num_tokens_per_channel, num_events_local_window,
                 num_events_end):
        super(PianoPrefixEndDataProcessor,
              self).__init__(embedding_size=embedding_size,
                             num_events=num_events,
                             num_tokens_per_channel=num_tokens_per_channel,
                             add_mask_token=True,
                             num_additional_tokens=2)
        # We need full access to the dataset and dataloader_generator
        self.dataloader_generator = dataloader_generator
        self.num_events_local_window = num_events_local_window
        self.num_events_end = num_events_end
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

            ======= ======
            before  after

        Returns:
        Sequences of the form
        (parenthesis for optional tokens depending on context):
        ======= ===== ===== === ===== ======= ====== === =====
        (after) (END) (PAD) SOD (PAD) (START) before END (PAD)
                              _____________
                              size max = 64

        """
        sequences_size = self.dataloader_generator.sequences_size
        batch_size, num_events, _ = x.size()
        assert num_events == sequences_size
        assert sequences_size > self.num_events_end + self.num_events_local_window

        x = cuda_variable(x.long())

        num_events_suffix = num_events_inpainted
        # the 2 accounts for the SOD and END tokens
        num_meta_events = 2
        max_num_events_suffix = sequences_size - (self.num_events_end +
                                                  num_meta_events) - 1
        if num_events_suffix is None:
            num_events_suffix = random.randint(self.num_events_local_window+1, max_num_events_suffix)
        else:
            assert (num_events_suffix > self.num_events_local_window
                    ), 'wrong number of events to be inpainted'

        # Slice x
        x = x[:, :self.num_events_end + num_events_suffix]
        batch_size, num_events, _ = x.size()

        # === Find end and start tokens in x
        is_start_token = (
            x[:, :,
              0] == self.start_tokens[0].unsqueeze(0).unsqueeze(0).repeat(
                  batch_size, num_events))
        is_end_token = (
            x[:, :, 0] == self.end_tokens[0].unsqueeze(0).unsqueeze(0).repeat(
                batch_size, num_events))
        contains_start_token = (is_start_token.sum(1) >= 1)
        contains_end_token = (is_end_token.sum(1) >= 1)
        # only one of those per sequence
        # Only valid when containes_end_token!!
        start_token_location = torch.argmax(is_start_token.long(), dim=1)
        end_token_location = torch.argmax(is_end_token.long(), dim=1)


        before = x[:, :num_events_suffix]
        after = x[:, num_events_suffix:num_events_suffix + self.num_events_end]
        placeholder_duration = self.dataloader_generator.get_elapsed_time(before)[:, -1]

        prefix_list, suffix_list = [], []
        # TODO batch this?!
        for (b, a, c_start_token, c_end_token, start_token_l,
             end_token_l) in zip(before, after, contains_start_token,
                                 contains_end_token, start_token_location,
                                 end_token_location):
            # assert START is not in end
            assert not torch.any(after[:, :, 0] == self.start_tokens[0]
                                 ), 'Start token located in after!'
            assert not (c_start_token and (start_token_l >= num_events_suffix)
                        ), "Start token located in after"
            # assert END is not in the first local_window tokens
            assert not (c_end_token and
                        (end_token_l < self.num_events_local_window)
                        ), "End token located in local_window"

            # Construction du prefix
            if c_end_token and (end_token_l < num_events_suffix):
                # END is in before
                prefix = torch.cat([
                    self.end_tokens.unsqueeze(0),
                    self.pad_tokens.unsqueeze(0).repeat(a.size(0) - 1, 1)
                ],
                                   dim=0)
            else:
                prefix = a

            ########################################################################
            # Safeguards, can be removed after a while
            if torch.any(prefix[:, 0] == self.start_tokens[0]):
                # START in after
                raise Exception
            if torch.any(prefix[:, 0] == self.pad_tokens[0]):
                # PADS in after: there needs to be an END
                # and they have to appear after END
                assert torch.any(prefix[:, 0] == self.end_tokens[0]
                                 ), 'after contains PADS, but no END'
                pads_locations = torch.where(
                    prefix[:, 0] == self.pad_tokens[0].unsqueeze(0))[0]
                end_location = torch.where(
                    prefix[:, 0] == self.end_tokens[0].unsqueeze(0))[0]
                assert (end_location.shape[0] == 1), 'several END in suffix'
                assert torch.all(
                    pads_locations > end_location), 'PADS before ENDS in after'
            ########################################################################

            # Construction du suffix
            if c_start_token and (start_token_l >=
                                  self.num_events_local_window):
                # START is in before, but not in the local window.
                # trim until START appears as the last element of the local window
                # (we don't want the model to predict START tokens)
                trim_begin = start_token_l - self.num_events_local_window + 1
                suffix = b[trim_begin:]
            elif c_end_token and (end_token_l < num_events_suffix):
                # END token is in before.
                # Remove END from suffix since it is appended later
                suffix = b[:end_token_l]
            else:
                suffix = b

            ########################################################################
            # Safeguards, can be removed after running the code for a while with no exception
            if torch.any(suffix[:, 0] == self.start_tokens[0]):
                # check START position
                start_location = torch.where(
                    suffix[:, 0] == self.start_tokens[0].unsqueeze(0))[0]
                assert (
                    start_location.shape[0] == 1), 'several STARTS in suffix'
                assert (start_location < self.num_events_local_window
                        ), 'START appears after local window'
            # No END in suffix (yet)
            is_end_token = torch.any(suffix[:, 0] == self.end_tokens[0])
            assert not is_end_token, 'end token in suffix before end token is added'
            ########################################################################

            # Now append end and pads
            num_events_pad_end = sequences_size - (
                self.num_events_end + len(suffix) + num_meta_events)
            assert num_events_pad_end > 0
            suffix = torch.cat([
                suffix,
                self.end_tokens.unsqueeze(0),
                self.pad_tokens.unsqueeze(0).repeat(num_events_pad_end, 1)
            ],
                               dim=0)

            assert len(prefix) + len(suffix) == sequences_size - 1
            prefix_list.append(prefix)
            suffix_list.append(suffix)

        prefix_tensor = torch.stack(prefix_list, dim=0)
        suffix_tensor = torch.stack(suffix_list, dim=0)
        sod = self.sod_symbols.unsqueeze(0).unsqueeze(0).repeat(
            batch_size, 1, 1)
        # creates final sequence
        y = torch.cat([prefix_tensor, sod, suffix_tensor], dim=1)

        # recompute padding mask
        _, num_events_output, _ = y.size()
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
        # add local windows, we only want "valid" local windows
        final_mask[:, :self.num_events_local_window, :] = True
        final_mask[:, self.num_events_end + 1:self.num_events_end + 1 +
                   self.num_events_local_window, :] = True

        # decoding_start and decoding_end
        decoding_start = self.num_events_end + self.num_events_local_window + 1
        # valid_suffix_len = torch.where(suffix_tensor[:, :, 0] == self.end_tokens[0])[0][0] + 1
        # decoding_end = (self.num_events_end + 1 + valid_suffix_len)

        # self.num_events_before + self.num_events_after + 1 is the location
        # of the SOD symbol (only the placeholder is added)
        metadata_dict = {
            'placeholder_duration': placeholder_duration,
            'decoding_start': decoding_start,
            'decoding_end': None,
            'original_sequence': y,
            'loss_mask': final_mask
        }
        return y, metadata_dict

    def compute_elapsed_time(self, metadata_dict):
        # Original sequence is in prefix order!
        x = metadata_dict['original_sequence']
        elapsed_time = self.dataloader_generator.get_elapsed_time(x)
        # add zero
        elapsed_time = torch.cat(
            [torch.zeros_like(elapsed_time)[:, :1], elapsed_time[:, :-1]],
            dim=1)

        # offset prefix
        elapsed_time[:, :self.num_events_end] = elapsed_time[:, :self.num_events_end] \
            + metadata_dict['placeholder_duration'].unsqueeze(1)
        elapsed_time[:, self.num_events_end:] = elapsed_time[:, self.num_events_end:] \
            - elapsed_time[:, self.num_events_end+1:self.num_events_end+2]

        assert not torch.any(elapsed_time < 0), 'Negative elapsed time'

        return elapsed_time

    def postprocess(self, x, decoding_end, metadata_dict):
        before = x[:, self.num_events_end+1:].to(self.end_tokens.device)

        # trim end
        num_events = before.shape[1]
        is_end_token = (
            before[:, :,
              0] == self.end_tokens[0].unsqueeze(0).unsqueeze(0).repeat(
                  1, num_events))
        contains_end_token = (is_end_token.sum(1) == 1)
        if contains_end_token:
            end_token_location = torch.argmax(is_end_token.long(), dim=1)
        else:
            raise Exception('no or more than 1 END token generated in suffix')
        before = before[:end_token_location]

        # trim start
        num_events = before.shape[1]
        is_start_token = (
            before[:, :,
              0] == self.start_tokens[0].unsqueeze(0).unsqueeze(0).repeat(
                  1, num_events))
        contains_start_token = (is_start_token.sum(1) == 1)
        if contains_start_token:
            start_token_location = torch.argmax(is_start_token.long(), dim=1)
            before = before[start_token_location+1:]

        after = x[:, :self.num_events_end].to(self.end_tokens.device)
        # trim end
        num_events = after.shape[1]
        is_end_token = (
            after[:, :,
              0] == self.end_tokens[0].unsqueeze(0).unsqueeze(0).repeat(
                  1, num_events))
        contains_end_token = (is_end_token.sum(1) == 1)
        if contains_end_token:
            end_token_location = torch.argmax(is_end_token.long(), dim=1)
            after = after[:end_token_location]

        # put all pieces in order:
        x_out = torch.cat([before, after], dim=1)
        return x_out
