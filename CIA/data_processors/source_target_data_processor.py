from .data_processor import DataProcessor
from torch import nn

class SourceTargetDataProcessor(nn.Module):
    """
    Abstract class used for preprocessing and embedding
    
    It basically contains two DataProcessors (one for the source and one for the target)
    Preprocessing: from ? -> (source, target, metadata_dict) where 
    - source is (batch_size, num_events_source, num_channels_source)
    - target is (batch_size, num_events_target, num_channels_target)
    - metadata_dict is a dictionnary of (batch_size, ...) tensors that can be used by positional encodings and first token.
    
    Embedding: from (batch_size, num_events, num_channels) ->
      (batch_size, num_events, num_channels, embedding_size)
    """

    def __init__(self,
                 encoder_data_processor: DataProcessor,
                 decoder_data_processor: DataProcessor):
        super(SourceTargetDataProcessor, self).__init__()
        self.encoder_data_processor = encoder_data_processor
        self.decoder_data_processor = decoder_data_processor
    
    @property
    def embedding_size_source(self):
        return self.encoder_data_processor.embedding_size
    
    @property
    def embedding_size_target(self):
        return self.decoder_data_processor.embedding_size
    
    @property
    def num_channels_source(self):
        return self.encoder_data_processor.num_channels
    
    @property
    def num_channels_target(self):
        return self.decoder_data_processor.num_channels
    
    @property
    def num_events_source(self):
        return self.encoder_data_processor.num_events
    
    @property
    def num_events_target(self):
        return self.decoder_data_processor.num_events
    
    @property
    def num_tokens_per_channel_target(self):
        return self.decoder_data_processor.num_tokens_per_channel
    
    def embed_source(self, x):
        """
        :param x: (..., num_channels)
        :return: (..., num_channels, embedding_size)
        """
        return self.encoder_data_processor.embed(x)

    def embed_target(self, x):
        """
        :param x: (..., num_channels)
        :return: (..., num_channels, embedding_size)
        """
        return self.decoder_data_processor.embed(x)

    def embed_step_source(self, x, channel_index):
        """
        :param x: (..., num_channels)
        :return: (..., num_channels, embedding_size)
        """
        return self.encoder_data_processor.embed_step(
            x, channel_index=channel_index)

    def embed_step_target(self, x, channel_index):
        """
        :param x: (..., num_channels)
        :return: (..., num_channels, embedding_size)
        """
        return self.decoder_data_processor.embed_step(
            x, channel_index=channel_index)
        
    def preprocess(self, x):
        """
        Subclasses must implement this method
                
        x comes directly from the data_loader_generator.
        source and target must be put on the GPUs if needed.
        :param x: ? 
        :return: (source, target, metadata_dict) 
        of size (batch_size, num_events_source, num_channels_source)
        (batch_size, num_events_target, num_channels_target)        
        """
        raise NotImplementedError
        

    