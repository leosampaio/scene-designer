import os
import glob

import numpy as np

import utils
from core.experiments import Experiment


class EmbeddingsForSBIR(Experiment):
    name = "embeddings-for-sbir"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            source_folder="/stornext/CVPNOBACKUP/scratch_4weeks/tb0035/projects/bumblebee/mturk2/src_des_data/",
            read_multiple_files=False,
            include_files_that_start_with="train_",
            exclude_files_that_end_with=".txt",
            specific_file="source.npz",
            batch_size=256
        )
        return hps

    def compute(self, model=None):

        if self.hps['read_multiple_files']:
            dataset_files = [f for f in glob.glob("{}/*".format(self.hps['source_folder']))
                             if os.path.basename(f).startswith(self.hps['include_files_that_start_with']) and
                             not os.path.basename(f).endswith(self.hps['exclude_files_that_end_with'])]
        else:
            dataset_files = [os.path.join(self.hps['source_folder'],
                                          self.hps['specific_file'])]

        # prepare saving folders
        embedding_out_dir = os.path.join(self.out_dir, 'embedding')
        if not os.path.isdir(embedding_out_dir):
            os.mkdir(embedding_out_dir)

        for dataset_file in dataset_files:
            z_data = []
            print("[Embedding] Preprocessing {}...".format(dataset_file))
            data = np.load(dataset_file, allow_pickle=True)
            x_data = data['data']
            x_data = model.dataset.preprocess_extra_sets_from_interp_experiment(x_data)

            # gather all z values
            print("[Embedding] Gathering embedding on {}...".format(dataset_file))
            for count, i in enumerate(range(0, len(x_data), self.hps['batch_size'])):
                end_idx = i + self.hps['batch_size'] if i + self.hps['batch_size'] < len(x_data) else len(x_data)
                batch_x = x_data[i:end_idx]
                results = model.predict_class(batch_x)
                z_data.append(results['embedding'])
                if count % 2 == 0:
                    print("[Embedding] Gathering embedding on {}... {}/{}".format(dataset_file, i, len(x_data)))

            z_data = np.concatenate(z_data, axis=0)

            print("[Embedding] Saving embedding for {}...".format(dataset_file))
            filepath = os.path.join(embedding_out_dir, '{}.npz'.format(os.path.basename(dataset_file)[:-3]))
            np.savez(filepath, embedding=z_data, ids=data['ids'], cat=data['cat'])
