from CIA.handlers.handler import Handler
from torch.distributed.distributed_c10d import get_world_size
from CIA.dataloaders.dataloader import DataloaderGenerator
from CIA.utils import all_reduce_scalar, dict_pretty_print, display_monitored_quantities, flatten, is_main_process, to_numpy, top_k_top_p_filtering
import torch
import os
from tqdm import tqdm
from itertools import islice
from datetime import datetime, time
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


class EncoderDecoderHandler(Handler):
    def __init__(self, model: DistributedDataParallel, model_dir: str,
                 dataloader_generator: DataloaderGenerator) -> None:
        super(EncoderDecoderHandler,
              self).__init__(model=model,
                             model_dir=model_dir,
                             dataloader_generator=dataloader_generator)

    # ==== Wrappers
    def forward(self, source, target, metadata_dict, h_pe_init=None):
        return self.model.forward(source,
                                  target,
                                  metadata_dict=metadata_dict,
                                  h_pe_init=h_pe_init)

    def forward_with_states(self,
                            memory,
                            target,
                            metadata_dict,
                            h_pe_init=None):
        return self.model.module.forward_memory_target_with_states(
            memory, target, metadata_dict=metadata_dict, h_pe_init=h_pe_init)

    def forward_step(self, memory, target, metadata_dict, state, i, h_pe):
        return self.model.module.forward_step(memory, target, metadata_dict,
                                              state, i, h_pe)

    def forward_source(self, source, metadata_dict):
        return self.model.module.forward_source(source, metadata_dict)

    def mask_source(self, source, masked_positions):
        return self.model.module.data_processor._mask_source(
            x=source, masked_positions=masked_positions)

    @property
    def num_tokens_per_channel_target(self):
        return self.model.module.data_processor.num_tokens_per_channel_target

    @property
    def num_channels_target(self):
        return self.model.module.data_processor.num_channels_target

    def load(self, early_stopped, recurrent):
        map_location = {'cuda:0': f'cuda:{dist.get_rank()}'}
        print(f'Loading models {self.__repr__()}')
        if early_stopped:
            print('Load early stopped model')
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            print('Load over-fitted model')
            model_dir = f'{self.model_dir}/overfitted'

        state_dict = torch.load(f'{model_dir}/model',
                                map_location=map_location)

        # copy transformer_with_states during inference
        if recurrent:
            transformer_with_states_dict = {}
            for k, v in state_dict.items():
                if 'transformer' in k:
                    new_key = k.replace('decoder.transformer',
                                        'decoder.transformer_with_states')
                    transformer_with_states_dict[new_key] = v
            state_dict.update(transformer_with_states_dict)

        self.model.load_state_dict(state_dict=state_dict)

    # ==== Training methods
    def epoch(
        self,
        data_loader,
        train=True,
        num_batches=None,
    ):
        means = None

        if train:
            self.train()
        else:
            self.eval()

        h_pe_init = None

        iterator = enumerate(islice(data_loader, num_batches))
        if is_main_process():
            iterator = tqdm(iterator, ncols=80)

        for sample_id, tensor_dict in iterator:

            # ==========================
            with torch.no_grad():
                x = tensor_dict['x']
                source, target, metadata_dict = self.data_processor.preprocess(
                    x)

            # ========Train decoder =============
            self.optimizer.zero_grad()
            forward_pass = self.forward(source=source,
                                        target=target,
                                        metadata_dict=metadata_dict,
                                        h_pe_init=h_pe_init)
            loss = forward_pass['loss']
            # h_pe_init = forward_pass['h_pe'].detach()

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                self.optimizer.step()

            # Monitored quantities
            monitored_quantities = forward_pass['monitored_quantities']

            # average quantities
            if means is None:
                means = {key: 0 for key in monitored_quantities}
            means = {
                key: value + means[key]
                for key, value in monitored_quantities.items()
            }

            del loss

        # renormalize monitored quantities
        for key, value in means.items():
            means[key] = all_reduce_scalar(value,
                                           average=True) / (sample_id + 1)

        # means = {
        #     key: all_reduce_scalar(value, average=True) / (sample_id + 1)
        #     for key, value in means.items()
        # }
        return means

    # ===== Generation methods
    def generate(self, source, metadata_dict, temperature, top_k=0, top_p=1.):
        """Generate using the EncoderDecoder conditionned on source

        Args:
            source (LongTensor): (batch_size, num_events_source, num_channels_source)
            temperature ([type]): [description]
            batch_size (int, optional): [description]. Defaults to 1.
            top_k (int, optional): [description]. Defaults to 0.
            top_p ([type], optional): [description]. Defaults to 1..

        Returns:
            [type]: [description]
        """
        assert self.recurrent
        self.eval()
        batch_size = source.size(0)

        # TODO hard coded value
        num_events = 1024

        # TODO URGENT ORIGINAL TOKEN IN METADATA_DICT

        x = torch.zeros(batch_size, num_events,
                        self.num_channels_target).long().to(source.device)
        with torch.no_grad():

            # init
            xi = torch.zeros_like(x)[:, 0, 0]
            state = None
            h_pe = None

            # compute memory only once
            memory = self.forward_source(source, metadata_dict)

            # i corresponds to the position of the token BEING generated
            for event_index in range(num_events):
                for channel_index in range(self.num_channels_target):
                    i = event_index * self.num_channels_target + channel_index

                    forward_pass = self.forward_step(
                        memory=memory,
                        target=xi,
                        metadata_dict=metadata_dict,
                        state=state,
                        i=i,
                        h_pe=h_pe)
                    weights = forward_pass['weights']

                    logits = weights / temperature

                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(logit,
                                                             top_k=top_k,
                                                             top_p=top_p)
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    for batch_index in range(batch_size):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel_target[channel_index]),
                                                           p=p[batch_index])

                        # TODO ADD switch
                        if metadata_dict['masked_positions'][
                                batch_index, event_index,
                                channel_index].item() == 0:
                            x[batch_index, event_index,
                              channel_index] = source[batch_index, event_index,
                              channel_index]
                        else:
                            x[batch_index, event_index,
                              channel_index] = int(new_pitch_index)

                    # update
                    xi = x[:, event_index, channel_index]
                    h_pe = forward_pass['h_pe']
                    state = forward_pass['state']

        # to score
        original_and_reconstruction = self.data_processor.postprocess(x.cpu())

        ###############################
        # Saving
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(f'{self.model_dir}/generations'):
            os.mkdir(f'{self.model_dir}/generations')

        # Write scores
        scores = []
        for k, tensor_score in enumerate(original_and_reconstruction):
            path_no_extension = f'{self.model_dir}/generations/{timestamp}_{k}'
            # TODO fix write signature
            scores.append(
                self.dataloader_generator.write(tensor_score,
                                                path_no_extension))
            # scores.append(self.dataloader_generator.write(tensor_score.unsqueeze(0),
            #                                               path_no_extension))
        ###############################

        return scores

    def generate_region(self,
                        source,
                        metadata_dict,
                        temperature,
                        start_event,
                        end_event,
                        top_k=0,
                        top_p=1.):
        """Generate using the EncoderDecoder conditionned on source

        Args:
            source (LongTensor): (batch_size, num_events_source, num_channels_source)
            temperature ([type]): [description]
            start_event:
            end_event:            
            batch_size (int, optional): [description]. Defaults to 1.
            top_k (int, optional): [description]. Defaults to 0.
            top_p ([type], optional): [description]. Defaults to 1..

        Returns:
            [type]: the inpainted region (batch_size, end_event - start_event, num_channels)
        """
        assert self.recurrent
        self.eval()
        batch_size = source.size(0)

        num_events = end_event - start_event

        x = torch.zeros(batch_size, num_events,
                        self.num_channels_target).long().to(source.device)
        with torch.no_grad():

            # init
            xi = torch.zeros_like(x)[:, 0, 0]
            state = None
            h_pe = None

            # compute memory only once
            memory = self.forward_source(source, metadata_dict)

            # i corresponds to the position of the token BEING generated
            for event_index in range(start_event, end_event):
                for channel_index in range(self.num_channels_target):
                    i = event_index * self.num_channels_target + channel_index

                    forward_pass = self.forward_step(
                        memory=memory,
                        target=xi,
                        metadata_dict=metadata_dict,
                        state=state,
                        i=i,
                        h_pe=h_pe)
                    weights = forward_pass['weights']

                    logits = weights / temperature

                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(logit,
                                                             top_k=top_k,
                                                             top_p=top_p)
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    for batch_index in range(batch_size):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel_target[channel_index]),
                                                           p=p[batch_index])
                        x[batch_index, event_index - start_event,
                          channel_index] = int(new_pitch_index)

                    # update
                    xi = x[:, event_index - start_event, channel_index]
                    h_pe = forward_pass['h_pe']
                    state = forward_pass['state']
        return x

    def inpaint(self, x, masked_positions, temperature=1.0, top_p=1., top_k=0):
        """Regenerated only tokens from x specified by masked_positions

        Args:
            x (LongTensor): (batch_size, num_events, num_channels)
            masked_positions (BoolTensor): same as x
        """
        source, _ = self.mask_source(x, masked_positions)
        metadata_dict = dict(original_sequence=x,
                             masked_positions=masked_positions)
        return self.generate(source,
                             metadata_dict,
                             temperature=temperature,
                             top_k=top_k,
                             top_p=top_p)

    def inpaint_region(self,
                       x,
                       start_event,
                       end_event,
                       masked_positions,
                       temperature=1.0,
                       top_p=1.,
                       top_k=0):
        """Regenerated only tokens from x specified by masked_positions between start_event and end_event

        Args:
            x (LongTensor): (batch_size, num_events, num_channels)
            masked_positions (BoolTensor): same as x
        """
        source, _ = self.mask_source(x, masked_positions)
        metadata_dict = dict(original_sequence=x,
                             masked_positions=masked_positions)
        region = self.generate_region(source,
                                      metadata_dict,
                                      start_event=start_event,
                                      end_event=end_event,
                                      temperature=temperature,
                                      top_k=top_k,
                                      top_p=top_p)

        output = torch.cat([x[:, :start_event], region, x[:, end_event:]],
                           dim=1)
        # TODO save here
        beginning = x[:, :start_event]
        return output, region, beginning

    def inpaint_region_optimized(self,
                                 x,
                                 start_event,
                                 end_event,
                                 masked_positions,
                                 temperature=1.0,
                                 top_p=1.,
                                 top_k=0):
        """Regenerated only tokens from x specified by masked_positions between start_event and end_event

        Args:
            x (LongTensor): (batch_size, num_events, num_channels)
            masked_positions (BoolTensor): same as x
        """
        self.eval()
        assert self.recurrent
        source, _ = self.mask_source(x, masked_positions)
        metadata_dict = dict(original_sequence=x,
                             masked_positions=masked_positions)

        with torch.no_grad():
            # compute state for autoregressive generation
            def extract_state_from_parallel_state(state_parallel, i):
                extracted_state = []
                for state in state_parallel:
                    # iterate on layers

                    # self attention
                    self_atn = state[0]
                    # extract -ith element on S and Z
                    self_atn_x = [self_atn[0][:, i], self_atn[1][:, i]]

                    # cross attention (DIAGONAL ONLY)
                    cross_atn = state[1]
                    cross_atn_x = cross_atn[i].item()

                    new_state = [self_atn_x, cross_atn_x]
                    extracted_state.append(new_state)
                return extracted_state

            # Init
            # compute memory only once
            memory = self.forward_source(source, metadata_dict)
            # Special case when start_event == 0
            # do not compute state in parallel
            if start_event == 0:
                state = None
                h_pe = None
                xi = torch.zeros_like(x)[:, 0, 0]
            else:
                weights, _, state_parallel = self.forward_with_states(
                    memory=memory, target=x, metadata_dict=metadata_dict)
                state = extract_state_from_parallel_state(
                    state_parallel, start_event * self.num_channels_target - 1)
                xi = x[:, start_event - 1, 3]

            # compute h_pe
            if start_event <= 1:
                h_pe = None
            else:
                target_embedded = self.model.module.data_processor.embed_target(
                    x)
                # add positional embeddings
                # Since h_pe is only updated every num_channel_target steps,
                # we can compute it at location self.num_channels_target *
                #  (start_event - 1)
                target_seq = flatten(
                    target_embedded)[:, :self.num_channels_target *
                                     (start_event - 1)]
                metadata_dict_sliced = {
                    k: v[:, :start_event - 1]
                    for k, v in metadata_dict.items()
                }
                _, h_pe = self.model.module.positional_embedding_target(
                    target_seq,
                    i=0,
                    h=None,
                    metadata_dict=metadata_dict_sliced)

            batch_size, num_events_target, num_channels_source = x.size()
            # print('WARNING: no excluded symbols')
            print('Warning: excluded symbols ON')
            # i corresponds to the position of the token BEING generated
            for event_index in range(start_event, end_event):
                for channel_index in range(self.num_channels_target):
                    i = event_index * self.num_channels_target + channel_index
                    # TODO skip computation if token is not masked

                    forward_pass = self.forward_step(
                        memory=memory,
                        target=xi,
                        metadata_dict=metadata_dict,
                        state=state,
                        i=i,
                        h_pe=h_pe)
                    weights = forward_pass['weights']

                    logits = weights / temperature

                    # TODO put this in a method so that it is applicable to all datasets
                    # exclude non note symbols:
                    exclude_symbols = ['START', 'END', 'XX']
                    for sym in exclude_symbols:
                        sym_index = self.dataloader_generator.dataset.value2index[
                            self.dataloader_generator.
                            features[channel_index]][sym]
                        logits[:, sym_index] = -np.inf

                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(logit,
                                                             top_k=top_k,
                                                             top_p=top_p)
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    for batch_index in range(batch_size):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel_target[channel_index]),
                                                           p=p[batch_index])
                        
                        # TODO regenerate or not depending on masked_positions?
                        x[batch_index, event_index,
                          channel_index] = int(new_pitch_index)

                    # update
                    xi = x[:, event_index, channel_index]
                    h_pe = forward_pass['h_pe']
                    state = forward_pass['state']
        return x.detach().cpu()

    def plot_attention(self, attentions_list, timestamp, name):
        """
        Helper function

        :param attentions_list: list of (batch_size, num_heads, num_tokens_encoder

        :return:
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        # to (batch_size, num_heads, num_tokens_decoder, num_tokens_encoder)
        attentions_batch = torch.cat([t.unsqueeze(2) for t in attentions_list],
                                     dim=2)

        # plot only batch 0 for now
        for batch_index, attentions in enumerate(attentions_batch):
            plt.clf()
            plt.cla()
            num_heads = attentions.size(0)
            for head_index, t in enumerate(attentions):
                plt.subplot(1, num_heads, head_index + 1)
                plt.title(f'Head {head_index}')
                mat = t.detach().cpu().numpy()
                sns.heatmap(mat, vmin=0, vmax=1, cmap="YlGnBu")
                plt.grid(True)
            plt.savefig(
                f'{self.model_dir}/generations/{timestamp}_{batch_index}_{name}.pdf'
            )
            # plt.show()
        plt.close()

    # TODO put this in data_processor/dataloader_generator
    # but hard!
    def init_generation_chorale(self, num_events, start_index):
        PAD = [
            d[PAD_SYMBOL]
            for d in self.dataloader_generator.dataset.note2index_dicts
        ]
        START = [
            d[START_SYMBOL]
            for d in self.dataloader_generator.dataset.note2index_dicts
        ]
        aa = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(
            1, start_index - 1, 1).long()
        bb = torch.Tensor(START).unsqueeze(0).unsqueeze(0).long()
        cc = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(
            1, num_events - start_index, 1).long()
        init_sequence = torch.cat([aa, bb, cc], 1)
        return init_sequence

    def compute_start_end_times(self, t, num_blocks, num_blocks_model):
        """

        :param t:
        :param num_blocks: num_blocks of the sequence to be generated
        :param num_blocks_model:
        :return:
        """
        # t_relative
        if num_blocks_model // 2 <= t < num_blocks - num_blocks_model // 2:
            t_relative = (num_blocks_model // 2)
        else:
            if t < num_blocks_model // 2:
                t_relative = t
            elif t >= num_blocks - num_blocks_model // 2:
                t_relative = num_blocks_model - (num_blocks - t)
            else:
                NotImplementedError

        # choose proper block to use
        t_begin = min(max(0, t - num_blocks_model // 2),
                      num_blocks - num_blocks_model)
        t_end = t_begin + num_blocks_model

        return t_begin, t_end, t_relative

    def generate_completion(
        self,
        num_completions,
        temperature,
        top_k,
        top_p,
        midi_file,
    ):
        """
        This method only works on harpsichord
        :param num_completions:
        :param temperature:
        :return:
        """
        self.eval()
        original = self.dataloader_generator.dataset.process_score(midi_file)
        original = self.dataloader_generator.dataset.tokenize(original)
        # TODO put in preprocess
        x = torch.stack([
            torch.LongTensor(original[e])
            for e in self.dataloader_generator.features
        ],
                        dim=-1)
        # todo filter
        # add batch_size
        num_events = 1024

        x = x.unsqueeze(0).repeat(num_completions, 1, 1)
        start_event_index = x.size(1)
        x = torch.cat([
            x,
            torch.zeros(num_completions, num_events - start_event_index,
                        self.num_channels_target).long()
        ],
                      dim=1)

        with torch.no_grad():
            # init
            xi = torch.zeros_like(x)[:, 0, 0]
            state = None
            h_pe = None

            # i corresponds to the position of the token BEING generated
            for event_index in range(num_events):
                for channel_index in range(self.num_channels_target):
                    i = event_index * self.num_channels_target + channel_index

                    forward_pass = self.forward_step(xi,
                                                     state=state,
                                                     i=i,
                                                     h_pe=h_pe)
                    weights = forward_pass['weights']

                    logits = weights / temperature

                    # # Removing these lines make the method applicable to all datasets
                    # TODO separate method in dataprocessor?
                    # # exclude non note symbols:
                    # exclude_symbols = ['START', 'END', 'XX']
                    # for sym in exclude_symbols:
                    #     sym_index = self.dataloader_generator.dataset.note2index_dicts[
                    #         channel_index][sym]
                    #     logits[:, sym_index] = -np.inf

                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(logit,
                                                             top_k=top_k,
                                                             top_p=top_p)
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    for batch_index in range(num_completions):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel_target[channel_index]),
                                                           p=p[batch_index])

                        # complete only
                        if event_index >= start_event_index:
                            x[batch_index, event_index,
                              channel_index] = int(new_pitch_index)

                    # update
                    xi = x[:, event_index, channel_index]
                    h_pe = forward_pass['h_pe']
                    state = forward_pass['state']

        # Saving
        original_and_reconstruction = self.data_processor.postprocess(x.cpu())
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(f'{self.model_dir}/completions'):
            os.mkdir(f'{self.model_dir}/completions')

        # Write scores
        scores = []
        for k, tensor_score in enumerate(original_and_reconstruction):
            path_no_extension = f'{self.model_dir}/completions/{timestamp}_{k}'
            scores.append(
                self.dataloader_generator.write(tensor_score,
                                                path_no_extension))

        return scores

    def test_decoder_with_states(self,
                                 source,
                                 metadata_dict,
                                 temperature,
                                 top_k=0,
                                 top_p=1.):
        """Generate using the EncoderDecoder conditionned on source

        Args:
            source (LongTensor): (batch_size, num_events_source, num_channels_source)
            temperature ([type]): [description]
            batch_size (int, optional): [description]. Defaults to 1.
            top_k (int, optional): [description]. Defaults to 0.
            top_p ([type], optional): [description]. Defaults to 1..

        Returns:
            [type]: [description]
        """
        assert self.recurrent
        self.eval()
        batch_size = source.size(0)

        import timeit
        start = timeit.default_timer()

        x = source.detach().to(source.device).clone()
        with torch.no_grad():
            # init

            state = None
            h_pe = None
            xi = torch.zeros_like(x)[:, 0, 0]

            # # compute memory only once
            memory = self.forward_source(source, metadata_dict)

            # compute state for autoregressive generation
            def extract_state_from_parallel_state(state_parallel, i):
                extracted_state = []
                for state in state_parallel:
                    # iterate on layers

                    # self attention
                    self_atn = state[0]
                    # extract -ith element on S and Z
                    self_atn_x = [self_atn[0][:, i], self_atn[1][:, i]]

                    # cross attention (DIAGONAL ONLY)
                    cross_atn = state[1]
                    cross_atn_x = cross_atn[i].item()

                    new_state = [self_atn_x, cross_atn_x]
                    extracted_state.append(new_state)
                return extracted_state

            weights, _, state_parallel = self.forward_with_states(
                memory=memory, target=x, metadata_dict=metadata_dict)

            # For inpainting between 100 and 200
            state = extract_state_from_parallel_state(state_parallel, 399)
            xi = x[:, 99, 3]

            # compute h_pe
            batch_size, num_events_target, num_channels_source = x.size()
            target_embedded = self.model.module.data_processor.embed_target(x)

            # add positional embeddings
            target_seq = flatten(target_embedded)[:, :4 * 100]
            metadata_dict_sliced = {
                k: v[:, :100]
                for k, v in metadata_dict.items()
            }
            _, h_pe = self.model.module.positional_embedding_target(
                target_seq, i=0, h=h_pe, metadata_dict=metadata_dict_sliced)
            
            # TODO PROBLEM HERE
            # index 400 being computed twice?!

            # i corresponds to the position of the token BEING generated
            for event_index in range(100, 150):
                for channel_index in range(self.num_channels_target):
                    i = event_index * self.num_channels_target + channel_index

                    forward_pass = self.forward_step(
                        memory=memory,
                        target=xi,
                        metadata_dict=metadata_dict,
                        state=state,
                        i=i,
                        h_pe=h_pe)
                    weights = forward_pass['weights']

                    logits = weights / temperature
                    
                    # TODO put this in a method so that it is applicable to all datasets
                    # exclude non note symbols:
                    # exclude_symbols = ['START', 'END', 'XX']
                    # for sym in exclude_symbols:
                    #     sym_index = self.dataloader_generator.dataset.value2index[
                    #         self.dataloader_generator.
                    #         features[channel_index]][sym]
                    #     logits[:, sym_index] = -np.inf

                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(logit,
                                                             top_k=top_k,
                                                             top_p=top_p)
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    for batch_index in range(batch_size):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel_target[channel_index]),
                                                           p=p[batch_index])
                        x[batch_index, event_index,
                          channel_index] = int(new_pitch_index)

                    # update
                    xi = x[:, event_index, channel_index]
                    h_pe = forward_pass['h_pe']
                    state = forward_pass['state']

        stop = timeit.default_timer()
        print(f'Time {stop - start}s')
        # add original
        x = torch.cat([source.cpu(), x.cpu()], dim=0)
        # to score
        original_and_reconstruction = self.data_processor.postprocess(x.cpu())

        ###############################
        # Saving
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(f'{self.model_dir}/generations'):
            os.mkdir(f'{self.model_dir}/generations')

        # Write scores
        scores = []
        for k, tensor_score in enumerate(original_and_reconstruction):
            path_no_extension = f'{self.model_dir}/generations/{timestamp}_{k}'
            scores.append(
                self.dataloader_generator.write(tensor_score,
                                                path_no_extension))
            # scores.append(self.dataloader_generator.write(tensor_score.unsqueeze(0),
            #                                               path_no_extension))
        ###############################

        return scores
