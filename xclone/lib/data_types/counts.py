from typing import Dict, Optional, Iterable, Union

import numpy as np
from scipy import sparse

class CountMatrix:
    """
    This class is a basic handler for the output of CellSNP
    It stores two matrices: AD and DP counts respectively.
    It supports row/column filtering and aggregation of values
    in row/column groups.
    """
    def __init__(
            self,
            rownames: np.array,
            colnames: np.array,
            ad_mtx: sparse.csc_matrix,
            dp_mtx: sparse.csc_matrix
        ):
        """
        :param rownames: np.array of N row names
        :param colnames: np.array of M column names
        :param ad_mtx: (N, M) matrix of AD counts
        :param dp_mtx: (N, M) matrix of DP counts
        """

        assert ad_mtx.shape == dp_mtx.shape,\
            "AD and DP matrices have different shapes:"\
            f" {ad_mtx.shape} and {dp_mtx.shape} respectively"
        assert ad_mtx.ndim == 2 and dp_mtx.ndim == 2, \
            "AD or DP matrix has wrong number of dimensions (must be 2):"\
            f" {ad_mtx.ndim} and {dp_mtx.ndim} respectively"
        assert type(ad_mtx) is type(dp_mtx),\
            "AD and DP matrices have different types:"\
            f" {type(ad_mtx)} and {type(dp_mtx)} respectively"
        assert rownames.size == ad_mtx.shape[0],\
            "Passed features don't align with count matrices:"\
            f" {rownames.size} passed, {ad_mtx.shape[0]} required"
        assert colnames.size == ad_mtx.shape[1],\
            "Passed barcodes don't align with count matrices:"\
            f" {colnames.size} passed, {ad_mtx.shape[1]} required"

        self.count_mx: Dict[str, type(ad_mtx)] = {"AD" : ad_mtx, "DP" : dp_mtx}
        self.rownames: np.array = np.array(list(rownames))
        self.colnames: np.array = np.array(list(colnames))

    @property
    def ad_mtx(self):
        """
        :return: AD count matrix
        """
        return self.count_mx["AD"]

    @property
    def dp_mtx(self):
        """
        :return: DP count matrix
        """
        return self.count_mx["DP"]

    # @property
    # def get_name(self):
    #     sorted_tag_ids = sorted(list(self.tags.keys()))
    #     return ''.join([
    #         f"[{tag_id}]({self.tags[tag_id]})"
    #         for tag_id in sorted_tag_ids
    #     ])

    def select_rows(self, rownames: Iterable):
        """
        This function implements feature-based filtering
        :param rownames: feature names
        :return: CountMatrix with features mentioned in rownames only
        """
        mask = np.isin(self.rownames, rownames)
        return CountMatrix(
            rownames=self.rownames[mask],
            colnames=self.colnames,
            ad_mtx=self.ad_mtx[mask, :],
            dp_mtx=self.dp_mtx[mask, :]
        )

    def select_cols(self, colnames: Iterable):
        """
        This function implements barcode-based filtering
        :param colnames: barcode names
        :return: CountMatrix with barcodes mentioned in rownames only
        """
        mask = np.isin(self.colnames, colnames)
        return CountMatrix(
            rownames=self.rownames,
            colnames=self.colnames[mask],
            ad_mtx=self.ad_mtx[:, mask],
            dp_mtx=self.dp_mtx[:, mask]
        )

    def select(
        self,
        rownames: Optional[Iterable] = None,
        colnames: Optional[Iterable] = None
    ):
        """
        Provides support of fancy indexing.
        With this function, one can subselect rows and columns
        of count matrix by providing row and column names.
        :param rownames: rows to select
        :param colnames: columns to select
        :return: Submatrix on the intersection of rownames and colnames
        """
        result = self
        if rownames is not None:
            result = result.select_rows(rownames)
        if colnames is not None:
            result = result.select_cols(colnames)
        return result


class SnpCountMatrix(CountMatrix):
    def __init__(
            self,
            snps: np.array,
            barcodes: np.array,
            ad_mtx: sparse.csc_matrix,
            dp_mtx: sparse.csc_matrix
        ):

        super().__init__(
            rownames=snps,
            colnames=barcodes,
            ad_mtx=ad_mtx,
            dp_mtx=dp_mtx
        )

    @classmethod
    def from_count_matrix(self, count_mtx: CountMatrix):
        self.super = CountMatrix(
            rownames=count_mtx.rownames,
            colnames=count_mtx.colnames,
            ad_mtx=count_mtx.ad_mtx,
            dp_mtx=count_mtx.dp_mtx
        )

    @property
    def snps(self):
        return self.rownames

    @property
    def barcodes(self):
        return self.colnames

    def select_snps(self, snps: Iterable):
        return SnpCountMatrix.from_count_matrix(
            super().select_cols(snps)
        )

    def select_barcodes(self, barcodes: Iterable):
        return SnpCountMatrix.from_count_matrix(
            super().select_rows(barcodes)
        )

    def select(self, snps: Optional[Iterable] = None, barcodes: Optional[Iterable] = None):
        return SnpCountMatrix.from_count_matrix(
            super().select(rownames=snps, colnames=barcodes)
        )

    def adjust_by_phasing(self, phasing):
        pass



class GeneCountMatrix(CountMatrix):
    def __init__(
            self,
            genes: np.array,
            barcodes: np.array,
            ad_mtx: Union[np.ndarray, sparse.csc_matrix],
            dp_mtx: Union[np.ndarray, sparse.csc_matrix]
        ):

        super().__init__(
            rownames=genes,
            colnames=barcodes,
            ad_mtx=ad_mtx,
            dp_mtx=dp_mtx
        )

    @classmethod
    def from_count_matrix(self, count_mtx: CountMatrix):
        self.super = CountMatrix(
            rownames=count_mtx.rownames,
            colnames=count_mtx.colnames,
            ad_mtx=count_mtx.ad_mtx,
            dp_mtx=count_mtx.dp_mtx
        )

    # @classmethod
    # def from_snp_matrix(self, snp_mtx: SnpCountMatrix, genome: Genome):
    #     self.from_count_matrix(snp_mtx)


    @property
    def genes(self):
        return self.rownames

    @property
    def barcodes(self):
        return self.colnames

    def select_genes(self, genes: Iterable):
        return GeneCountMatrix.from_count_matrix(
            super().select_cols(genes)
        )

    def select_barcodes(self, barcodes: Iterable):
        return GeneCountMatrix.from_count_matrix(
            super().select_rows(barcodes)
        )

    def select(self, genes: Optional[Iterable] = None, barcodes: Optional[Iterable] = None):
        return GeneCountMatrix.from_count_matrix(
            super().select(rownames=genes, colnames=barcodes)
        )
