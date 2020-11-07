import functools
import multiprocessing as mp
import operator
import os
import pickle
from typing import List

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from data_model.dataset import Table, Cell, Entity
from datasets import DatasetEnum
from generators import CandidateGenerator, EmbeddingCandidateGenerator
from utils.functions import chunk_list


class CEAAnnotator:
    def __init__(self,
                 generator: CandidateGenerator,
                 micro_table_size: int = -1,
                 threads: int = mp.cpu_count()):
        """
        :param generator:
        :param micro_table_size:
        :param threads: max number of threads used to parallelize the annotation
        """
        assert threads > 0
        self._generator = generator
        self._threads = threads
        self._micro_table_size = micro_table_size

    @property
    def generator_id(self):
        return self._generator.id

    def annotate_table(self, table: Table, target_cells: List[Cell], filename=None):
        # check existing result, if filename is given
        if filename and os.path.isfile(filename):
            return pickle.load(open(filename, 'rb'))

        # keep the cell-search_key pair -> results may be shuffled!
        search_key_cell_dict = {table.get_search_key(cell_): cell_ for cell_ in target_cells}

        # Parallelize: if there are many cells, annotate chunks of cells (like mini-tables)
        # TODO delegate parallelization to Generators
        if self._micro_table_size > 0:
            # print("Parallel", table, len(target_cells))
            results = functools.reduce(operator.iconcat,
                                       process_map(self._generator.select_candidates,
                                                   list(chunk_list(list(search_key_cell_dict.keys()),
                                                                   self._micro_table_size)),
                                                   max_workers=2),
                                       [])
        else:
            # print("NO Parallel", table, len(target_cells))
            results = self._generator.select_candidates(list(search_key_cell_dict.keys()))

        for search_key, candidates in results:
            if candidates:
                table.annotate_cell(search_key_cell_dict[search_key], Entity(candidates[0]))  # first candidate = best

        if filename:
            pickle.dump(table, open(filename, 'wb'))

        return table

    def annotate_dataset(self, dataset: DatasetEnum, n: int = None):
        """
        Annotate tables of a given CEA dataset.
        :param dataset: CEA dataset to annotate
        :param n: if given, only the first n tables are annotated

        :return:
        """
        targets = dataset.get_target_cells()
        tables = []
        tables_filenames = []
        target_cells = []
        annotated_tables = []
        print(*self._generator.id, dataset.name)
        for table in list(dataset.get_tables())[:n]:
            filename = os.path.join(
                os.path.dirname(__file__),
                'annotations',
                '%s_%s_%s_%s_%s.pkl' % (*self._generator.id, dataset.name, table.tab_id))
            if table.tab_id in targets:
                tables.append(table)
                target_cells.append(targets[table.tab_id])
                tables_filenames.append(filename)

        threads = self._threads if self._micro_table_size <= 0 else int(self._threads / 2)
        # Do not parallelize CUDA executions
        if isinstance(self._generator, EmbeddingCandidateGenerator):
            new_annotated_tables = []
            for t, tc, tf in tqdm(zip(tables, target_cells, tables_filenames), total=len(tables)):
                new_annotated_tables.append(self.annotate_table(t, tc, tf))
        else:  # Parallelize: 1 table per process
            new_annotated_tables = process_map(self.annotate_table,
                                               tables, target_cells, tables_filenames,
                                               max_workers=threads)

        # for ann_table in new_annotated_tables:
        #     filename = os.path.join(
        #         os.path.dirname(__file__),
        #         'annotations',
        #         '%s_%s_%s_%s_%s.pkl' % (*self._generator.id, dataset.name, ann_table.tab_id))
        #     pickle.dump(ann_table, open(filename, 'wb'))

        return annotated_tables + new_annotated_tables
