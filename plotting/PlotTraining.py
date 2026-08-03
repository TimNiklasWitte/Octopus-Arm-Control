from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():

    algorithm_name_list = ["DDPG", "DDPG_HER", "PPO", "SAC", "SAC_HER"]

    df_list = []
    for algorithm_name in algorithm_name_list:
        log_dir = f"../{algorithm_name}/logs"
        if algorithm_name == "SAC_HER":
            log_dir = f"../{algorithm_name}/logs_5"

        df = load_dataframe(log_dir)
        df = df.loc[:, ["avg number steps to goal", "avg min distance", "success rate"]]

        df = df.loc[:1000]
        df["algorithm"] = algorithm_name

        df_list.append(df)
    
    df = pd.concat(df_list)


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    sns.lineplot(data=df,
                x="step",
                y="avg number steps to goal",
                hue="algorithm",
                alpha=0.6,
                ax=axes[0])

    sns.lineplot(data=df,
                x="step",
                y="avg min distance",
                hue="algorithm",
                alpha=0.6,
                ax=axes[1])

    sns.lineplot(data=df,
                x="step",
                y="success rate",
                hue="algorithm",
                alpha=0.6,
                ax=axes[2])

    for ax in axes:
        ax.grid(True)

    axes[0].set_title("avg number steps to goal")
    axes[1].set_title("avg min distance")
    axes[2].set_title("success rate")

    
    for ax in axes:
        ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=True)


    plt.tight_layout(rect=[0, 0.12, 1, 1])


    plt.savefig("./plots/Training_metrics.png", dpi=200)
   

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")