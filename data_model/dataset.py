import urllib.parse
from abc import ABC
from typing import List, Dict, Tuple, Iterable, NamedTuple
from dataclasses import dataclass, field

import pandas as pd
from pandas.errors import ParserError

from data_model.lookup import SearchKey


class Cell(NamedTuple):
    row_id: int
    col_id: int

    @property
    def id(self) -> Tuple[int, int]:
        return self.row_id, self.col_id


class Column(NamedTuple):
    col_id: int


class Entity(NamedTuple):
    uri: str

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Entity):
            return urllib.parse.unquote(self.uri).lower() == urllib.parse.unquote(o.uri).lower()
        return False


class Class(NamedTuple):
    uri: str


class CellAnnotation(NamedTuple):
    cell: Cell
    entities: List[Entity]

    def __eq__(self, o: object) -> bool:
        if isinstance(o, CellAnnotation):
            return len(set(self.entities) & set(o.entities)) > 0
        return False


class ColumnAnnotation(NamedTuple):
    column: Column
    classes: List[Class]


class AbstractTable(ABC):
    def __init__(self, tab_id: str):
        self._tab_id = tab_id
        self._cell_annotations: Dict[Cell, CellAnnotation] = {}
        self._column_annotations: Dict[Column, ColumnAnnotation] = {}

    @property
    def tab_id(self) -> str:
        return self._tab_id

    @property
    def cell_annotations(self) -> Dict[Cell, CellAnnotation]:
        return self._cell_annotations

    @property
    def column_annotations(self) -> Dict[Column, ColumnAnnotation]:
        return self._column_annotations

    def get_cells(self) -> List[Cell]:
        return list(self._cell_annotations)

    def get_columns(self) -> List[Column]:
        return list(self._column_annotations)


class Table(AbstractTable):
    def __init__(self, tab_id: str, filepath: str):
        super().__init__(tab_id)
        try:
            self._df = pd.read_csv(filepath, header=None, keep_default_na=False, escapechar='\\', dtype=str)
            self._gap = 0
        except ParserError:
            # Some tables are not VALID CSVs because of:
            # - a title row (not commented out), e.g., %C3%93scar_Rivas#0.csv in CEA_Round2
            # - a wrong header, e.g., 10th_Canadian_Parliament#1.csv in CEA_Round2
            self._df = pd.read_csv(filepath, header=None, keep_default_na=False, escapechar='\\', dtype=str, skiprows=1)
            self._gap = 1
        self._df = self._df.applymap(lambda x: x.replace('""', ''))  # Pandas missplaces quotes in quoted fields

    def annotate_cell(self, cell: Cell, entity: Entity):
        self._cell_annotations[cell] = CellAnnotation(cell, [entity])

    def annotate_column(self, column: Column, class_: Class):
        self._column_annotations[column] = ColumnAnnotation(column, [class_])

    def get_row(self, row_id: int):
        return self._df.iloc[row_id - self._gap]

    def get_column(self, col_id: int):
        return self._df[col_id]

    # def get_label(self, row_id: int, col_id: int) -> str:
    #     return self.get_row(row_id)[col_id]
    #
    # def get_context(self, row_id: int, col_ids: List[int]):
    #     return self.get_row(row_id)[col_ids]

    def get_search_key(self, cell: Cell) -> SearchKey:
        row = self.get_row(cell.row_id)
        return SearchKey(row[cell.col_id],
                         row[self._df.columns.drop(cell.col_id)].to_dict())


class GTTable(AbstractTable):

    def set_cell_annotations(self, triples: Iterable[Tuple[int, int, str]]):
        """
        Utility method that sets all the annotations for the GT
        :param triples: a list of tuples (row_id, col_id, entities_list)
        :return:
        """
        self._cell_annotations = {cell_annotation.cell: cell_annotation
                                  for cell_annotation in [CellAnnotation(Cell(triple[0], triple[1]),  # cell(row, col)
                                                                         [Entity(e) for e in triple[2]])  # entities
                                                          for triple in triples]}

    # def annotate_cell(self, cell: Cell, entities: List[Entity]):
    #     self._cell_annotations[cell] = CellAnnotation(cell, entities)
    #
    # def annotate_column(self, column: Column, classes: List[Class]):
    #     self._column_annotations[column] = ColumnAnnotation(column, classes)
