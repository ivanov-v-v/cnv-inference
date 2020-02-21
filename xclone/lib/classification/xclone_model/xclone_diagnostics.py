from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import classification.xclone_model.xclone_routines as xclone_routines

class GTConvergenceTracker:
    def __init__(self):
        self._ground_truth_negloglik = None

    def compute_true_negloglik(self, xclone_instance):
        new_params = xclone_instance._params.copy()
        new_params.I_RNA = new_params.I_DNA
        xclone_instance._sampler.update_H_RNA(new_params)
        xclone_instance._sampler.update_posterior(new_params)
        self._ground_truth_negloglik = -new_params.total_loglikelihood()

    def make_snapshot(self, xclone_instance):
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


    def perform_diagnostics(self, xclone_instance, window_width=500):
        history = xclone_instance._history
        curr_iter = xclone_instance._iter_count

        if self._ground_truth_negloglik is None:
            self.compute_true_negloglik(xclone_instance)
        
        clear_output()
        fig, axes = plt.subplots(
            1, 3, figsize=(24, 8),
            sharex=True
        )
        fig.suptitle(f"Last <={window_width} iterations (out of {xclone_instance._iter_count})")
        xgrid = np.arange(curr_iter + 1)[-window_width:]

        tracked_entities = ["accuracy", "changed", "negloglik"]
        titles = [f"Accuracy (in %). Max: {np.round(np.max(xclone_instance._history['accuracy']), 2)}",
                  "% of reassigned labels",
                  "Total negative loglikelihood"]
        ylabels = ["%", "%", "value"]

        for i in range(3):
            ax = axes[i]
            ax.set_title(titles[i])
            ax.set_xlabel("iteration")
            ax.set_ylabel(ylabels[i])

            for update_iter in history["updates"]:
                if update_iter >= curr_iter - window_width:
                    ax.axvline(update_iter, color="black", alpha=0.4, linestyle="--")

            if tracked_entities[i] == "negloglik":
                ax.axhline(np.min(history["negloglik"]), color="green", label="best assignment", linestyle="--")
                ax.axhline(self._ground_truth_negloglik, color="black", label="ground truth", linestyle="--")
            else:
                ax.axhline(history[tracked_entities[i]][np.argmax(history["negloglik"])], color="green", linestyle="--", label="best assignment")
                if tracked_entities[i] == "accuracy":
                    ax.axhline(np.max(history["accuracy"]), color="red", linestyle="--", label="maximal")

            yvals = np.array(history[tracked_entities[i]])[-window_width:]
            sns.scatterplot(xgrid, yvals, color="black", ax=axes[i], label="observations", alpha=0.3)
            #ewma = np.ravel(pd.DataFrame(yvals).ewm(span=50).mean().values)
            running_1st_quartile = np.ravel(pd.DataFrame(yvals).expanding().quantile(0.25).values)
            running_median = np.ravel(pd.DataFrame(yvals).expanding().median().values)
            running_3rd_quartile = np.ravel(pd.DataFrame(yvals).expanding().quantile(0.75).values)
#            print(xgrid.shape, yvals.shape, running_1st_quartile.shape, running_median.shape, running_3rd_quartile.shape)
            sns.lineplot(xgrid, running_median, color="red", ax=ax, label="running median")
            ax.fill_between(xgrid, running_1st_quartile, running_3rd_quartile, color="xkcd:tangerine", alpha=0.4, label="running iterquartile range")
            ax.legend().get_frame().set_facecolor("white")

        fig.savefig(f"{xclone_instance._figures_dir}"
                    f"/convergence_{curr_iter}.png")
        plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f"Probabilities of clonal label assignments. Iteration {curr_iter}")
        param_dict = xclone_instance._params.__dict__
        titles = ["DNA", "", "RNA"]

        cmap = "coolwarm"#sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
        assignment_probas = {
            modality : np.vstack([
                xclone_routines.compute_assignment_probas(
                    cell_id,
                    A=param_dict[f"A_{modality}"],
                    R=param_dict[f"R_{modality}"],
                    X_CLONE=param_dict["X_CLONE"],
                    logbincoeffs=param_dict[f"logbincoeff_{modality}"],
                    f=param_dict["f"]
                ) for cell_id in range(param_dict[f"M_{modality}"])
            ]) for modality in ["DNA", "RNA"]
        }
        vmin = np.min([assignment_probas[modality] for modality in ["DNA", "RNA"]])
        vmax = np.max([assignment_probas[modality] for modality in ["DNA", "RNA"]])
        for i in [0, 2]:
            ax = axes[i]
            modality = titles[i]
            ax.set_title(modality)
            ax.set_xlabel("clonal label")
            ax.set_ylabel("cell")

            assert np.allclose(assignment_probas[modality].sum(axis=1), 1),\
                f"Assignment probabilities don't add to 1 for {modality}!"
            # print(assignment_probas)
            sns.heatmap(assignment_probas[modality], ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
            
        axes[1].set_title("GROUND_TRUTH")
        axes[1].set_xlabel("clonal label")
        axes[1].set_ylabel("cell")
        one_hot = np.zeros(shape=(param_dict["M_DNA"], param_dict["K"]))
        for i in range(param_dict["M_DNA"]):
            one_hot[i, param_dict["I_DNA"][i]] = 1
        sns.heatmap(
            one_hot, 
            ax=axes[1], 
            cmap=sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True), 
            vmin=0, vmax=1
        )

        fig.savefig(f"{xclone_instance._figures_dir}"
                    f"/probas_{curr_iter}.png")
        plt.show()

        print(xclone_instance._params.total_loglikelihood(return_addends=True))
        if len(history["updates"]) > 0 and (curr_iter == history["updates"][-1]):
            with open(f"{xclone_instance._report_dir}/accuracy.txt", "a+") as callback_file:
                callback_file.write(
                    f"{curr_iter}\t"
                    f"{round(history['accuracy'][-1], 4)}\n"
                )
                print(f"Accuracy: {round(history['accuracy'][-1], 4)}%")
            with open(xclone_instance._log_path, "w") as log_file:
                log_file.write(
                    "{}\t{:.3f}\t{:.3f}\n".format(
                        curr_iter,
                        history['negloglik'][-1],
                        history['changed'][-1]
                   )
                )
                log_file.flush()
            print(
                "Iteration {} — labelling update!"
                "{:.2f} --> {:.2f}, {:.0f}% labels reassigned".format(
                    curr_iter,
                    np.min(history["negloglik"][:-1]) if curr_iter > 1 else -xclone_instance._init_loglik,
                    np.min(history["negloglik"][-1]),
                    history["changed"][-1],
                ),
                file=xclone_instance._log_stream
            )
            most_likely_labels = np.array([
                np.argmax(xclone_routines.compute_assignment_probas(
                    cell_id,
                    A=param_dict[f"A_RNA"],
                    R=param_dict[f"R_RNA"],
                    X_CLONE=param_dict["X_CLONE"],
                    logbincoeffs=param_dict[f"logbincoeff_RNA"],
                    f=param_dict["f"]
                )) for cell_id in range(param_dict[f"M_RNA"])
            ])
            print("Accuracy if the most probable labels are assigned: {:.2f}%".format(
                100 * np.mean(most_likely_labels == param_dict["I_DNA"])
            ))
