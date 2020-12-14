import multiprocessing as mp
import os
import pickle
import gc

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from data_model.dataset import Table, Entity
from datasets import DatasetEnum
from generators import EmbeddingCandidateGenerator, Generator, HybridGenerator, HybridGeneratorSimulator
from pathlib import Path


class CEAAnnotator:
    def __init__(self,
                 generator: Generator,
                 max_workers: int = mp.cpu_count()):
        """
        :param generator:
        :param max_workers: max number of workers used to parallelize the annotation
        """
        assert max_workers > 0
        self._generator = generator
        self._max_workers = max_workers

    @property
    def generator_id(self):
        return self._generator.id

    def annotate_table(self, table: Table):

        folder_path = os.path.join(os.path.dirname(__file__),
                                   'annotations',
                                   table.dataset_id,
                                   self._generator.id)
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(folder_path, '%s.pkl' % table.tab_id)

        # check existing result
        if not os.path.exists(filename):

            # keep the cell-search_key pair -> results may be shuffled!
            search_key_cell_dict = table.get_search_keys_cells_dict()

            if isinstance(self._generator, HybridGenerator):
                # try to reuse already annotated tables
                tables = []
                for generator in self._generator.generators:
                    subfolder_path = os.path.join(os.path.dirname(__file__),
                                                  'annotations',
                                                  table.dataset_id,
                                                  generator.id)
                    subfilename = os.path.join(subfolder_path, '%s.pkl' % table.tab_id)

                    if not os.path.exists(subfilename):
                        break

                    tables.append(pickle.load(open(subfilename, 'rb')))

                if len(tables) == len(self._generator.generators):
                    results = HybridGeneratorSimulator.get_candidates(*tables)  # combine results from many tables
                else:
                    results = self._generator.get_candidates(table)

            else:
                results = self._generator.get_candidates(table)

            for search_key, candidates in results:
                if candidates:
                    for cell in search_key_cell_dict[search_key]:
                        table.annotate_cell(cell, Entity(candidates[0]))  # first candidate = best

            pickle.dump(table, open(filename, 'wb'))

            gc.collect()

        return pickle.load(open(filename, 'rb'))

    def annotate_dataset(self, dataset: DatasetEnum):
        """
        Annotate tables of a given CEA dataset.
        :param dataset: CEA dataset to annotate
        :return:
        """
        print(self._generator.id, dataset.name)

        tables = dataset.get_tables()
        total_tables = dataset.total_tables()
        if self._max_workers == 1:  # Do not parallelize
            new_annotated_tables = []
            for table in tqdm(tables, total=total_tables):
                new_annotated_tables.append(self.annotate_table(table))
        else:  # Parallelize: 1 table per process
            new_annotated_tables = process_map(self.annotate_table,
                                               tables,
                                               max_workers=self._max_workers,
                                               total=total_tables)

        return new_annotated_tables
