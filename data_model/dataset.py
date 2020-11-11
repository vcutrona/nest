from typing import List, Dict, Tuple, Iterable, NamedTuple

import pandas as pd
from pandas.errors import ParserError

from data_model.kgs import Entity, Class, Property
from data_model.lookup import SearchKey


class Cell(NamedTuple):
    row_id: int
    col_id: int

    @property
    def id(self) -> Tuple[int, int]:
        return self.row_id, self.col_id


class Column(NamedTuple):
    col_id: int


class ColumnRelation(NamedTuple):
    source: int
    target: int

    @property
    def id(self) -> Tuple[int, int]:
        return self.source, self.target


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


class PropertyAnnotation(NamedTuple):
    related_columns: ColumnRelation
    properties: List[Property]


class Table:
    def __init__(self, tab_id: str, dataset_id: str, csv_path: str):
        # ids
        self._tab_id = tab_id
        self._dataset_id = dataset_id

        # cell values
        try:
            self._df = pd.read_csv(csv_path, header=None, keep_default_na=False, escapechar='\\', dtype=str)
            self._gap = 0
        except ParserError:
            # Some tables are not VALID CSVs because of:
            # - a title row (not commented out), e.g., %C3%93scar_Rivas#0.csv in CEA_Round2
            # - a wrong header, e.g., 10th_Canadian_Parliament#1.csv in CEA_Round2
            self._df = pd.read_csv(csv_path, header=None, keep_default_na=False, escapechar='\\', dtype=str, skiprows=1)
            self._gap = 1
        self._df = self._df.applymap(lambda x: x.replace('""', ''))  # Pandas missplaces quotes in quoted fields

        # annotations (predictions)
        self._cell_annotations: Dict[Cell, CellAnnotation] = {}
        self._column_annotations: Dict[Column, ColumnAnnotation] = {}
        self._property_annotations: Dict[ColumnRelation, PropertyAnnotation] = {}
        # annotations (gt)
        self._gt_cell_annotations: Dict[Cell, CellAnnotation] = {}
        self._gt_column_annotations: Dict[Column, ColumnAnnotation] = {}
        self._gt_property_annotations: Dict[ColumnRelation, PropertyAnnotation] = {}

    @property
    def tab_id(self) -> str:
        return self._tab_id

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def cell_annotations(self) -> Dict[Cell, CellAnnotation]:
        return self._cell_annotations

    @property
    def column_annotations(self) -> Dict[Column, ColumnAnnotation]:
        return self._column_annotations

    @property
    def property_annotations(self) -> Dict[ColumnRelation, PropertyAnnotation]:
        return self._property_annotations

    @property
    def gt_cell_annotations(self) -> Dict[Cell, CellAnnotation]:
        return self._gt_cell_annotations

    @property
    def gt_column_annotations(self) -> Dict[Column, ColumnAnnotation]:
        return self._gt_column_annotations

    @property
    def gt_property_annotations(self) -> Dict[ColumnRelation, PropertyAnnotation]:
        return self._gt_property_annotations

    def get_row(self, row_id: int):
        return self._df.iloc[row_id - self._gap]

    def get_column(self, col_id: int):
        return self._df[col_id]

    def get_search_key(self, cell: Cell) -> SearchKey:
        row = self.get_row(cell.row_id)
        return SearchKey(row[cell.col_id],
                         tuple(row[self._df.columns.drop(cell.col_id)].to_dict().items()))

    def get_search_keys_cells_dict(self) -> Dict[SearchKey, List[Cell]]:
        search_keys_dict = {}
        for cell in self._gt_cell_annotations:
            search_key = self.get_search_key(cell)
            if search_key not in search_keys_dict:
                search_keys_dict[search_key] = []
            search_keys_dict[search_key].append(cell)
        return search_keys_dict

    def get_annotated_cells(self) -> List[Cell]:
        return list(self._cell_annotations)

    def get_gt_cells(self) -> List[Cell]:
        return list(self._gt_cell_annotations)

    def get_annotated_columns(self) -> List[Column]:
        return list(self._column_annotations)

    def get_gt_columns(self) -> List[Column]:
        return list(self._gt_column_annotations)

    def get_annotated_properties(self) -> List[ColumnRelation]:
        return list(self._property_annotations)

    def get_gt_properties(self) -> List[ColumnRelation]:
        return list(self._gt_property_annotations)

    def annotate_cell(self, cell: Cell, entity: Entity):
        self._cell_annotations[cell] = CellAnnotation(cell, [entity])

    def annotate_column(self, column: Column, class_: Class):
        self._column_annotations[column] = ColumnAnnotation(column, [class_])

    def annotate_property(self, related_columns: ColumnRelation, property_: Property):
        self._property_annotations[related_columns] = PropertyAnnotation(related_columns, [property_])

    def set_gt_cell_annotations(self, triples: Iterable[Tuple[int, int, List[str]]]):
        """
        Utility method that sets all the cell annotations for the GT
        :param triples: a list of tuples (row_id, col_id, List(entities_list))
        :return:
        """
        self._gt_cell_annotations = {cell_annotation.cell: cell_annotation
                                     for cell_annotation in [CellAnnotation(Cell(triple[0], triple[1]),
                                                                            [Entity(e) for e in triple[2]])
                                                             for triple in triples]}

    def set_gt_column_annotations(self, pairs: Iterable[Tuple[int, List[str]]]):
        """
        Utility method that sets all the column annotations for the GT
        :param pairs: a list of pairs (col_id, List(types_list))
        :return:
        """
        self._gt_column_annotations = {col_annotation.column: col_annotation
                                       for col_annotation in [ColumnAnnotation(Column(pair[0]),
                                                                               [Class(e) for e in pair[1]])
                                                              for pair in pairs]}

    def set_gt_property_annotations(self, triples: Iterable[Tuple[int, int, List[str]]]):
        """
        Utility method that sets all the property annotations for the GT
        :param triples: a list of tuples (source_col_id, target_col_id, List(properties_list))
        :return:
        """
        self._gt_property_annotations = {prop_annotation.related_columns: prop_annotation
                                         for prop_annotation in
                                         [PropertyAnnotation(ColumnRelation(triple[0], triple[1]),
                                                             [Property(e) for e in triple[2]])
                                          for triple in triples]}

# class OldTable:
#     def __init__(self, tab_id: str, dataset: DatasetEnum, filepath: str):
#         super().__init__(tab_id, dataset)
#         try:
#             self._df = pd.read_csv(filepath, header=None, keep_default_na=False, escapechar='\\', dtype=str)
#             self._gap = 0
#         except ParserError:
#             # Some tables are not VALID CSVs because of:
#             # - a title row (not commented out), e.g., %C3%93scar_Rivas#0.csv in CEA_Round2
#             # - a wrong header, e.g., 10th_Canadian_Parliament#1.csv in CEA_Round2
#             self._df = pd.read_csv(filepath, header=None, keep_default_na=False, escapechar='\\', dtype=str, skiprows=1)
#             self._gap = 1
#         self._df = self._df.applymap(lambda x: x.replace('""', ''))  # Pandas missplaces quotes in quoted fields
#
#     def annotate_cell(self, cell: Cell, entity: Entity):
#         self._cell_annotations[cell] = CellAnnotation(cell, [entity])
#
#     def annotate_column(self, column: Column, class_: Class):
#         self._column_annotations[column] = ColumnAnnotation(column, [class_])
#
#     def set_column_annotations(self, column_annotations):
#         self._column_annotations = column_annotations
#
#     def annotate_property(self, related_columns: ColumnRelation, property_: Property):
#         self._property_annotations[related_columns] = PropertyAnnotation(related_columns, [property_])
#
#     def set_property_annotations(self, property_annotations):
#         self._property_annotations = property_annotations
#
#     def get_row(self, row_id: int):
#         return self._df.iloc[row_id - self._gap]
#
#     def get_column(self, col_id: int):
#         return self._df[col_id]
#
#     # def get_label(self, row_id: int, col_id: int) -> str:
#     #     return self.get_row(row_id)[col_id]
#     #
#     # def get_context(self, row_id: int, col_ids: List[int]):
#     #     return self.get_row(row_id)[col_ids]
#
#     def get_search_key(self, cell: Cell) -> SearchKey:
#         row = self.get_row(cell.row_id)
#         return SearchKey(row[cell.col_id],
#                          tuple(row[self._df.columns.drop(cell.col_id)].to_dict().items()))
#
#
# class OldGTTable:
#
#     def set_cell_annotations(self, triples: Iterable[Tuple[int, int, List[str]]]):
#         """
#         Utility method that sets all the cell annotations for the GT
#         :param triples: a list of tuples (row_id, col_id, List(entities_list))
#         :return:
#         """
#         self._cell_annotations = {cell_annotation.cell: cell_annotation
#                                   for cell_annotation in [CellAnnotation(Cell(triple[0], triple[1]),
#                                                                          [Entity(e) for e in triple[2]])
#                                                           for triple in triples]}
#
#     def set_column_annotations(self, pairs: Iterable[Tuple[int, List[str]]]):
#         """
#         Utility method that sets all the column annotations for the GT
#         :param pairs: a list of pairs (col_id, List(types_list))
#         :return:
#         """
#         self._column_annotations = {col_annotation.column: col_annotation
#                                     for col_annotation in [ColumnAnnotation(Column(pair[0]),
#                                                                             [Class(e) for e in pair[1]])
#                                                            for pair in pairs]}
#
#     def set_property_annotations(self, triples: Iterable[Tuple[int, int, List[str]]]):
#         """
#         Utility method that sets all the property annotations for the GT
#         :param triples: a list of tuples (source_col_id, target_col_id, List(properties_list))
#         :return:
#         """
#         self._property_annotations = {prop_annotation.related_columns: prop_annotation
#                                       for prop_annotation in [PropertyAnnotation(ColumnRelation(triple[0], triple[1]),
#                                                                                  [Property(e) for e in triple[2]])
#                                                               for triple in triples]}

# def annotate_cell(self, cell: Cell, entities: List[Entity]):
#     self._cell_annotations[cell] = CellAnnotation(cell, entities)
#
# def annotate_column(self, column: Column, classes: List[Class]):
#     self._column_annotations[column] = ColumnAnnotation(column, classes)
