# system utilities
import sys

class SimulationManager:
    def __init__(
            self,
            classifier,
            sampler,
            verbose: bool = False
    ):
        self.classifier = classifier
        self.sampler = sampler
        self._logfile = sys.stdout if verbose else open(os.devnull, "w+")

    def run_simulations(self):
        sampled_counts_df, y_true = self.sampler.sample()
        print("Done with sampling", file=self._logfile)
        y_pred = self.classifier.predict(sampled_counts_df)
        print("Done with predicting", file=self._logfile)
        na_stats = sampled_counts_df.isna().mean(axis=0)
        n_rows = sampled_counts_df.shape[0]
        return y_pred, y_true, na_stats, n_rows