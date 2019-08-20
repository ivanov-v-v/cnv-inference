class CountMatrix:
    def __init__(
            self,
            features,
            barcodes,
            ad_df,
            dp_df,
            tags=None
        ): # types?

        # are these tests enough?
        assert ad_df.shape == dp_df.shape
        assert features.size == ad_df.shape[0]
        assert barcodes.size == ad_df.shape[1]

        self.ad_df = ad_df # to sparse?
        self.dp_df = dp_df # to sparse?
        self.row_names = features
        self.col_names = barcodes
        self.tags = tags
