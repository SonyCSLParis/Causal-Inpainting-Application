from DatasetManager.chorale_dataset import ChoraleBeatsDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata

from CIA.dataloaders.dataloader import DataloaderGenerator

subdivision = 4
num_voices = 4
metadatas = [
    FermataMetadata(),
    TickMetadata(subdivision=subdivision),
    KeyMetadata()
]


class BachDataloaderGenerator(DataloaderGenerator):
    def __init__(self, sequences_size):
        dataset_manager = DatasetManager()

        chorale_dataset_kwargs = {
            'voice_ids':      [0, 1, 2, 3],
            'metadatas':      metadatas,
            'sequences_size': sequences_size,
            'subdivision':    subdivision,
        }

        dataset: ChoraleBeatsDataset = dataset_manager.get_dataset(
            name='bach_chorales_beats',
            **chorale_dataset_kwargs
        )
        super(BachDataloaderGenerator, self).__init__(dataset=dataset)
        # self.features = ['Soprano', 'Alto', 'Tenor', 'Bass']
        self.features = list(range(4))
    
    # Warning different meanings for sequence size:
    # in ChoraleBeatsDataset it's the number of beats,
    # here it's the number of events (i.e. in sixteenth notes)
    @property
    def sequences_size(self):
        return self.dataset.sequences_size * self.dataset.subdivision
    
    @property
    def num_channels(self):
        return 4


    def dataloaders(self, batch_size, num_workers=0, shuffle_train=True,
                    shuffle_val=False):
        # discard metadata
        # and put num_channels (num_voices) at the last dimension
        return [({'x': t[0].transpose(1, 2)}
                 for t in dataloader)
                for dataloader
                in self.dataset.data_loaders(batch_size, num_workers=num_workers,
                                             shuffle_train=shuffle_train,
                                             shuffle_val=shuffle_val
                                             )]

    def write(self, x, path):
        score = self.dataset.tensor_to_score(x.transpose(1, 0))
        score.write('xml', f'{path}.xml')
        return score

    def to_score(self, tensor_score):
        score = self.dataset.tensor_to_score(
            tensor_score.transpose(1, 0)
        )
        return score
