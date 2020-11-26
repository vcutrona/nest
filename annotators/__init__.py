import multiprocessing as mp
import os
import pickle

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from data_model.dataset import Table, Entity
from datasets import DatasetEnum
from generators import EmbeddingCandidateGenerator, Generator


class CEAAnnotator:
    def __init__(self,
                 generator: Generator,
                 threads: int = 6):  # mp.cpu_count()):
        """
        :param generator:
        :param threads: max number of threads used to parallelize the annotation
        """
        assert threads > 0
        self._generator = generator
        self._threads = threads

    @property
    def generator_id(self):
        return self._generator.id

    def annotate_table(self, table: Table):
        filename = os.path.join(
                os.path.dirname(__file__),
                'annotations',
                '%s_%s_%s.pkl' % (self._generator.id, table.dataset_id, table.tab_id))

        # check existing result
        if not os.path.exists(filename):
            # keep the cell-search_key pair -> results may be shuffled!
            search_key_cell_dict = table.get_search_keys_cells_dict()

            # Parallelize: if there are many cells, annotate chunks of cells (like mini-tables)
            # # TODO delegate parallelization to Generators
            # if self._micro_table_size > 0:
            #     # print("Parallel", table, len(target_cells))
            #     results = functools.reduce(operator.iconcat,
            #                                process_map(self._generator.get_candidates,
            #                                            list(chunk_list(list(search_key_cell_dict.keys()),
            #                                                            self._micro_table_size)),
            #                                            max_workers=2),
            #                                [])
            # else:
            #     # print("NO Parallel", table, len(target_cells))

            results = self._generator.get_candidates(table)
            for search_key, candidates in results:
                if candidates:
                    for cell in search_key_cell_dict[search_key]:
                        table.annotate_cell(cell, Entity(candidates[0]))  # first candidate = best

            pickle.dump(table, open(filename, 'wb'))

        return pickle.load(open(filename, 'rb'))

    def annotate_dataset(self, dataset: DatasetEnum):
        """
        Annotate tables of a given CEA dataset.
        :param dataset: CEA dataset to annotate
        :return:
        """
        # targets = dataset.get_target_cells()
        # tables_filenames = []
        # target_cells = []
        print(self._generator.id, dataset.name)
        # for table in list(dataset.get_tables())[:n]:
        #     filename = os.path.join(
        #         os.path.dirname(__file__),
        #         'annotations',
        #         '%s_%s_%s_%s_%s.pkl' % (*self._generator.id, dataset.name, table.tab_id))
        #     if table.tab_id in targets:
        #         tables.append(table)
        #         target_cells.append(targets[table.tab_id])
        #         tables_filenames.append(filename)

        # threads = self._threads if self._micro_table_size <= 0 else int(self._threads / 2)
        tables = list(dataset.get_tables())
        # Do not parallelize CUDA executions
        if isinstance(self._generator, EmbeddingCandidateGenerator):
            new_annotated_tables = []
            for table in tqdm(tables):
                new_annotated_tables.append(self.annotate_table(table))
        else:  # Parallelize: 1 table per process
            new_annotated_tables = process_map(self.annotate_table,
                                               tables,
                                               max_workers=self._threads)

        # for ann_table in new_annotated_tables:
        #     filename = os.path.join(
        #         os.path.dirname(__file__),
        #         'annotations',
        #         '%s_%s_%s_%s_%s.pkl' % (*self._generator.id, dataset.name, ann_table.tab_id))
        #     pickle.dump(ann_table, open(filename, 'wb'))

        return new_annotated_tables
