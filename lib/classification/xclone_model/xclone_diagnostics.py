from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import classification.xclone_model.xclone_routines as xclone_routines

def gt_callback(xclone_instance):
    with open(f"{xclone_instance._report_dir}/accuracy.txt", "a+") \
            as callback_file:
        accuracy = np.mean(xclone_instance._params.I_DNA
                           == xclone_instance._params.I_RNA)
        callback_file.write(
            f"{xclone_instance._iter_count}\t"
            f"{round(accuracy, 4)}\n"
        )
        print(f"Accuracy: {round(100 * accuracy, 4)}%")


class GTConvergenceTracker:
    def __init__(self):
        self._ground_truth_negloglik = None

    def compute_true_negloglik(self, xclone_instance):
        new_params = xclone_instance._params.copy()
        new_params.I_RNA = new_params.I_DNA
        xclone_instance._sampler.update_H_RNA(new_params)
        xclone_instance._sampler.update_posterior_ASR(new_params)
        self._ground_truth_negloglik = -new_params.total_loglikelihood()

    def update_history(self, xclone_instance):
        xclone_instance._history["accuracy"].append(
            100 * np.mean(xclone_instance._candidate_params.I_DNA
                          == xclone_instance._candidate_params.I_RNA)
        )
        xclone_instance._history["changed"].append(
            100 * np.mean(xclone_instance._params.I_RNA
                          != xclone_instance._candidate_params.I_RNA)
        )
        xclone_instance._history["negloglik"].append(
            -xclone_instance._candidate_loglik
        )

    def display_diagnostics(self, xclone_instance, window_width=500):
        if self._ground_truth_negloglik is None:
            self.compute_true_negloglik(xclone_instance)

        clear_output()
        fig, axes = plt.subplots(
            1, 3, figsize=(24, 8),
            sharex=True
        )
        fig.suptitle(f"Last <={window_width} iterations (out of {xclone_instance._iter_count})")
        xgrid = np.arange(xclone_instance._iter_count + 1)[-window_width:]

        tracked_entities = ["accuracy", "changed", "negloglik"]
        titles = ["Accuracy (in %)", "% of reassigned labels", "Total negative loglikelihood"]
        ylabels = ["%", "%", "value"]

        for i in range(3):
            ax = axes[i]
            ax.set_title(titles[i])
            ax.set_xlabel("iteration")
            ax.set_ylabel(ylabels[i])

            for update_iter in xclone_instance._history["updates"]:
                if update_iter >= xclone_instance._iter_count - window_width:
                    ax.axvline(update_iter, color="black", alpha=0.4, linestyle="--")

            if tracked_entities[i] == "negloglik":
                ax.axhline(-xclone_instance._best_loglik, color="green", label="current best", linestyle="--")
                ax.axhline(self._ground_truth_negloglik, color="black", label="ground_truth", linestyle="--")

            yvals = np.array(xclone_instance._history[tracked_entities[i]])[-window_width:]
            sns.scatterplot(xgrid, yvals, color="black", ax=axes[i], label="observations", alpha=0.3)
            ewma = np.ravel(pd.DataFrame(yvals).ewm(span=50).mean().values)
            sns.lineplot(xgrid, ewma, color="red", ax=ax, label="ewma (span=50)")
            ax.legend().get_frame().set_facecolor("white")

        fig.savefig(f"{xclone_instance._figures_dir}"
                    f"/convergence_{xclone_instance._iter_count}.png")
        plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle("Probabilities of clonal label assignments")
        param_dict = xclone_instance._params.__dict__
        titles = ["DNA", "", "RNA"]

        cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
        for i in [0, 2]:
            ax = axes[i]
            modality = titles[i]
            ax.set_title(modality)
            ax.set_xlabel("clonal label")
            ax.set_ylabel("cell")
            assignment_probas = np.vstack([
                xclone_routines.compute_assignment_probas(
                    cell_id,
                    A=param_dict[f"A_{modality}"],
                    R=param_dict[f"R_{modality}"],
                    X_CLONE=param_dict["X_CLONE"],
                    logbincoeffs=param_dict[f"logbincoeff_{modality}"],
                    f=param_dict["f"]
                ) for cell_id in range(param_dict[f"M_{modality}"])
            ])
            assert np.allclose(assignment_probas.sum(axis=1), 1),\
                f"Assignment probabilities don't add to 1 for {modality}!"
            # print(assignment_probas)
            sns.heatmap(assignment_probas, ax=ax, cmap=cmap, vmin=0, vmax=1)
            
        axes[1].set_title("GROUND_TRUTH")
        axes[1].set_xlabel("clonal label")
        axes[1].set_ylabel("cell")
        one_hot = np.zeros(shape=(param_dict["M_DNA"], param_dict["K"]))
        for i in range(param_dict["M_DNA"]):
            one_hot[i, param_dict["I_DNA"][i]] = 1
        sns.heatmap(one_hot, ax=axes[1], cmap=cmap, vmin=0, vmax=1)

        fig.savefig(f"{xclone_instance._figures_dir}"
                    f"/probas_{xclone_instance._iter_count}.png")
        plt.show()

        print(xclone_instance._params.total_loglikelihood(return_addends=True))
