from CIA.handlers.handler import Handler
from torch.distributed.distributed_c10d import get_world_size
from CIA.dataloaders.dataloader import DataloaderGenerator
from CIA.utils import all_reduce_scalar, dict_pretty_print, display_monitored_quantities, flatten, is_main_process, to_numpy, top_k_top_p_filtering
import torch
import os
from tqdm import tqdm
from itertools import islice
from datetime import datetime
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


class DecoderPrefixHandler(Handler):
    def __init__(self, model: DistributedDataParallel, model_dir: str,
                 dataloader_generator: DataloaderGenerator) -> None:
        super().__init__(model=model,
                         model_dir=model_dir,
                         dataloader_generator=dataloader_generator)

    def plot(self, epoch_id, monitored_quantities_train,
             monitored_quantities_val) -> None:
        if is_main_process():
            for k, v in monitored_quantities_train.items():
                self.writer.add_scalar(f'{k}/train', v, epoch_id)
            for k, v in monitored_quantities_val.items():
                self.writer.add_scalar(f'{k}/val', v, epoch_id)

    def forward_with_states(self,
                            target,
                            metadata_dict,
                            h_pe_init=None):
        return self.model.module.forward_with_states(
            target, metadata_dict=metadata_dict, h_pe_init=h_pe_init)

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

        # if recurrent, we must also load the "with_states" version
        if recurrent:
            transformer_with_states_dict = {}
            for k, v in state_dict.items():
                if 'transformer' in k:
                    new_key = k.replace('transformer.transformer', 'transformer.transformer_with_states')
                    transformer_with_states_dict[new_key] = v
            state_dict.update(transformer_with_states_dict)
        self.model.load_state_dict(
            state_dict=state_dict
            )


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
                x, metadata_dict = self.data_processor.preprocess(x)

            # ========Train decoder =============
            self.optimizer.zero_grad()
            forward_pass = self.forward(target=x,
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

    def inpaint_non_optimized(self, x, temperature=1., top_p=1., top_k=0):
        # TODO add arguments to preprocess
        true_x = x.clone()
        x, metadata_dict =  self.data_processor.preprocess(x)
        original_x = x.clone()
        print(f'Placeholder duration: {metadata_dict["placeholder_duration"]}')
        assert self.recurrent
        self.eval()
        batch_size, num_events, _ = x.size()

        # TODO only works with batch_size=1 at present
        assert x.size(0) == 1

        decoding_end = None
        decoding_start_event = metadata_dict['decoding_start']
        with torch.no_grad():
            # init
            xi = torch.zeros_like(x)[:, 0, 0]
            state = None
            h_pe = None

            # i corresponds to the position of the token BEING generated
            for event_index in range(0, num_events):
                for channel_index in range(self.num_channels_target):
                    i = event_index * self.num_channels_target + channel_index

                    forward_pass = self.forward_step(
                        xi,
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
                        if event_index >= decoding_start_event:
                            new_pitch_index = np.random.choice(np.arange(
                                self.num_tokens_per_channel_target[channel_index]),
                                                            p=p[batch_index])
                            x[batch_index, event_index,
                            channel_index] = int(new_pitch_index)

                            end_symbol_index = self.dataloader_generator.dataset.value2index[
                            self.dataloader_generator.features[channel_index]]['END']
                            if end_symbol_index == int(new_pitch_index):
                                decoding_end = event_index


                    # update
                    xi = x[:, event_index, channel_index]
                    h_pe = forward_pass['h_pe']
                    state = forward_pass['state']

                if decoding_end is not None:
                    break

        print(f'Num events_generated: {decoding_end - decoding_start_event}')
        # to score
        original_and_reconstruction = self.data_processor.postprocess(
            x.cpu(),
            decoding_end,
            metadata_dict
            )

        # find decoding end for original sequence
        end_symbol_index = self.dataloader_generator.dataset.value2index[
                            self.dataloader_generator.features[0]]['END']
        for i in range(decoding_start_event, num_events):
            if original_x[0, i, 0].item() == end_symbol_index:
                decoding_end = i
                break

        original_x = self.data_processor.postprocess(
            original_x.cpu(),
            decoding_end,
            metadata_dict
        )

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


        for k, tensor_score in enumerate(original_x):
            path_no_extension = f'{self.model_dir}/generations/{timestamp}_{k}_original'
            # TODO fix write signature
            scores.append(
                self.dataloader_generator.write(tensor_score,
                                                path_no_extension))

        for k, tensor_score in enumerate(true_x):
            path_no_extension = f'{self.model_dir}/generations/{timestamp}_{k}_true'
            # TODO fix write signature
            scores.append(
                self.dataloader_generator.write(tensor_score,
                                                path_no_extension))

        ###############################

        return scores

    def test_inpaint(self, x, temperature=1., top_p=1., top_k=0):
        # TODO add arguments to preprocess
        true_x = x.clone()
        x, metadata_dict =  self.data_processor.preprocess(x)
        original_x = x.clone()
        print(f'Placeholder duration: {metadata_dict["placeholder_duration"]}')
        assert self.recurrent
        self.eval()
        batch_size, num_events, _ = x.size()

        # TODO only works with batch_size=1 at present
        assert x.size(0) == 1

        decoding_end = None
        decoding_start_event = metadata_dict['decoding_start']
        with torch.no_grad():
            def extract_state_from_parallel_state(state_parallel, i):
                extracted_state = []
                for state in state_parallel:
                    # iterate on layers
                    # extract -ith element on S and Z
                    self_atn_x = [state[0][:, i], state[1][:, i]]
                    extracted_state.append(self_atn_x)
                return extracted_state

            # init:
            # compute state
            _, _, state_parallel = self.forward_with_states(target=x, metadata_dict=metadata_dict)
            state = extract_state_from_parallel_state(
                state_parallel=state_parallel,
                i=decoding_start_event * self.num_channels_target - 1
            )
            xi = x[:, decoding_start_event - 1, self.num_channels_target - 1]
            target_embedded = self.model.module.data_processor.embed(
                    x)

            # compute positional embeddings, i.e. h_pe
            # TODO -1 HERE ?!
            # TODO COMPUTE EXACTLY! NEEDS TO PERFORM FORWARD_STEPS
            target_seq = flatten(
                target_embedded)[:, :self.num_channels_target *
                                    (decoding_start_event - 1)]

            metadata_dict_sliced = metadata_dict.copy()
            # AND -1 HERE!
            metadata_dict_sliced['original_sequence'] = metadata_dict['original_sequence'][:, :decoding_start_event - 1]
            _, h_pe = self.model.module.positional_embedding(
                target_seq,
                i=0,
                h=None,
                metadata_dict=metadata_dict_sliced)

            for channel_index in range(self.num_channels_target - 1):
                original_token = x[:, decoding_start_event - 1, channel_index]
                original_token_embedded = self.model.module.data_processor.embed_step(original_token, channel_index)
                metadata_dict['original_token'] = original_token
                _, h_pe = self.model.module.positional_embedding.forward_step(
                    original_token_embedded,
                    i=(decoding_start_event -1) * self.num_channels_target + channel_index,
                    h=h_pe,
                    metadata_dict=metadata_dict
                )

            # i corresponds to the position of the token BEING generated
            for event_index in range(decoding_start_event, num_events):
                for channel_index in range(self.num_channels_target):
                    i = event_index * self.num_channels_target + channel_index
                    print(f'{i} : {h_pe}')

                    forward_pass = self.forward_step(
                        xi,
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
                        if event_index >= decoding_start_event:
                            new_pitch_index = np.random.choice(np.arange(
                                self.num_tokens_per_channel_target[channel_index]),
                                                            p=p[batch_index])
                            x[batch_index, event_index,
                            channel_index] = int(new_pitch_index)

                            end_symbol_index = self.dataloader_generator.dataset.value2index[
                            self.dataloader_generator.features[channel_index]]['END']
                            if end_symbol_index == int(new_pitch_index):
                                decoding_end = event_index


                    # update
                    xi = x[:, event_index, channel_index]
                    h_pe = forward_pass['h_pe']
                    state = forward_pass['state']

                if decoding_end is not None:
                    break

        print(f'Num events_generated: {decoding_end - decoding_start_event}')
        # to score
        original_and_reconstruction = self.data_processor.postprocess(
            x.cpu(),
            decoding_end,
            metadata_dict
            )

        # find decoding end for original sequence
        end_symbol_index = self.dataloader_generator.dataset.value2index[
                            self.dataloader_generator.features[0]]['END']
        for i in range(decoding_start_event, num_events):
            if original_x[0, i, 0].item() == end_symbol_index:
                decoding_end = i
                break

        original_x = self.data_processor.postprocess(
            original_x.cpu(),
            decoding_end,
            metadata_dict
        )

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


        for k, tensor_score in enumerate(original_x):
            path_no_extension = f'{self.model_dir}/generations/{timestamp}_{k}_original'
            # TODO fix write signature
            scores.append(
                self.dataloader_generator.write(tensor_score,
                                                path_no_extension))

        for k, tensor_score in enumerate(true_x):
            path_no_extension = f'{self.model_dir}/generations/{timestamp}_{k}_true'
            # TODO fix write signature
            scores.append(
                self.dataloader_generator.write(tensor_score,
                                                path_no_extension))

        ###############################

        return scores


    def inpaint(self, x,
                metadata_dict,
                num_max_generated_events=None,
                temperature=1., top_p=1., top_k=0):
        """x is already in the right "prefix" format

        Args:
            x ([type]): [description]
            metadata_dict ([type]): [description]
            num_max_generated_events ([type], optional): [description]. Defaults to None.
            temperature ([type], optional): [description]. Defaults to 1..
            top_p ([type], optional): [description]. Defaults to 1..
            top_k (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """
        # original_x = x.clone()

        print(f'Placeholder duration: {metadata_dict["placeholder_duration"]}')

        assert self.recurrent
        self.eval()
        batch_size, num_events, _ = x.size()


        # TODO only works with batch_size=1 at present
        assert x.size(0) == 1

        decoding_end = None
        decoding_start_event = metadata_dict['decoding_start']
        if num_max_generated_events is None:
            decoding_end_event = num_events
        else:
            decoding_end_event = decoding_start_event + num_max_generated_events


        with torch.no_grad():
            # def extract_state_from_parallel_state(state_parallel, i):
            #     extracted_state = []
            #     for state in state_parallel:
            #         # iterate on layers
            #         # extract -ith element on S and Z
            #         self_atn_x = [state[0][:, i], state[1][:, i]]
            #         extracted_state.append(self_atn_x)
            #     return extracted_state

            # # init:
            # # compute state
            # _, _, state_parallel = self.forward_with_states(target=x, metadata_dict=metadata_dict)
            # state = extract_state_from_parallel_state(
            #     state_parallel=state_parallel,
            #     i=decoding_start_event * self.num_channels_target - 1
            # )

            # compute_state slices using :state_index AFTER adding the dummy symbol before x
            # state index is thus the state which is used to predict the x token located at state_index
            state = self.model.module.compute_state(target=x,
                                       metadata_dict=metadata_dict,
                                       state_index=decoding_start_event * self.num_channels_target - 1
                                       )

            xi = x[:, decoding_start_event - 1, self.num_channels_target - 1]
            target_embedded = self.model.module.data_processor.embed(
                    x)

            # compute positional embeddings, i.e. h_pe
            # TODO -1 HERE ?!
            # TODO COMPUTE EXACTLY! NEEDS TO PERFORM FORWARD_STEPS
            target_seq = flatten(
                target_embedded)[:, :self.num_channels_target *
                                    (decoding_start_event - 1)]

            metadata_dict_sliced = metadata_dict.copy()
            # AND -1 HERE!
            metadata_dict_sliced['original_sequence'] = metadata_dict['original_sequence'][:, :decoding_start_event - 1]
            _, h_pe = self.model.module.positional_embedding(
                target_seq,
                i=0,
                h=None,
                metadata_dict=metadata_dict_sliced)

            for channel_index in range(self.num_channels_target - 1):
                original_token = x[:, decoding_start_event - 1, channel_index]
                original_token_embedded = self.model.module.data_processor.embed_step(original_token, channel_index)
                metadata_dict['original_token'] = original_token
                _, h_pe = self.model.module.positional_embedding.forward_step(
                    original_token_embedded,
                    i=(decoding_start_event -1) * self.num_channels_target + channel_index,
                    h=h_pe,
                    metadata_dict=metadata_dict
                )

            # i corresponds to the position of the token BEING generated
            for event_index in range(decoding_start_event, decoding_end_event):
                for channel_index in range(self.num_channels_target):
                    i = event_index * self.num_channels_target + channel_index
                    print(f'{i} : {h_pe}')

                    forward_pass = self.forward_step(
                        xi,
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
                        if event_index >= decoding_start_event:
                            new_pitch_index = np.random.choice(np.arange(
                                self.num_tokens_per_channel_target[channel_index]),
                                                            p=p[batch_index])
                            x[batch_index, event_index,
                            channel_index] = int(new_pitch_index)

                            end_symbol_index = self.dataloader_generator.dataset.value2index[
                            self.dataloader_generator.features[channel_index]]['END']
                            if end_symbol_index == int(new_pitch_index):
                                decoding_end = event_index


                    # update
                    xi = x[:, event_index, channel_index]
                    h_pe = forward_pass['h_pe']
                    state = forward_pass['state']

                if decoding_end is not None:
                    break


        if decoding_end is None:
            done = False
            decoding_end = decoding_end_event
        else:
            done = True

        x_inpainted = self.data_processor.postprocess(
            x.cpu(),
            decoding_end,
            metadata_dict
            )

        generated_region = x[:, decoding_start_event:decoding_end]
        return x_inpainted, generated_region, done



    # ===== Generation methods
    def generate_non_recurrent(self,
                               temperature,
                               batch_size=1,
                               plot_attentions=False,
                               top_k=0,
                               top_p=1.):
        self.eval()

        with torch.no_grad():
            x = self.init_generation(num_events=self.data_processor.num_events)
            # Duplicate along batch dimension
            x = x.repeat(batch_size, 1, 1)

            h_pe_init = None

            for event_index in range(x.size(1)):
                # for event_index in range(self.data_processor.num_events):
                for channel_index in range(self.num_channels_target):
                    forward_pass = self.forward(x, h_pe_init=h_pe_init)

                    weights_per_voice = forward_pass['weights_per_category']
                    weights = weights_per_voice[channel_index]

                    # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
                    logits = weights[:, event_index, :] / temperature

                    # # exclude non note symbols:
                    exclude_symbols = ['START', 'END', 'XX']
                    for sym in exclude_symbols:
                        sym_index = self.dataloader_generator.dataset.note2index_dicts[
                            channel_index][sym]
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
                            self.num_tokens_per_channel[channel_index]),
                                                           p=p[batch_index])
                        x[batch_index, event_index,
                          channel_index] = int(new_pitch_index)

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
        ###############################

        return scores

    def generate(self, metadata_dict, temperature, batch_size=1, top_k=0, top_p=1.):
        assert self.recurrent
        self.eval()
        # num_events = 4 * 4 * 24
        # num_events = 240
        # TODO hardcoded
        num_events = 1024

        x = torch.zeros(batch_size, num_events,
                        self.num_channels_target).long()

        # needed for sos_embedding
        metadata_dict['original_sequence'] = x
        with torch.no_grad():
            # init
            xi = torch.zeros_like(x)[:, 0, 0]
            state = None
            h_pe = None

            # i corresponds to the position of the token BEING generated
            for event_index in range(num_events):
                for channel_index in range(self.num_channels_target):
                    i = event_index * self.num_channels_target + channel_index

                    forward_pass = self.forward_step(
                        xi,
                        metadata_dict=metadata_dict,
                        state=state,
                        i=i,
                        h_pe=h_pe)
                    weights = forward_pass['weights']

                    logits = weights / temperature

                    # # Removing these lines make the method applicable to all datasets
                    # TODO separate method in dataprocessor?
                    # # exclude non note symbols:
                    # exclude_symbols = ['START', 'END', 'XX']
                    exclude_symbols = ['END', 'XX']
                    for sym in exclude_symbols:
                        # TODO unify
                        # sym_index = self.dataloader_generator.dataset.note2index_dicts[
                        #     channel_index][sym]
                        sym_index = self.dataloader_generator.dataset.value2index[
                            self.dataloader_generator.features[channel_index]][sym]
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
