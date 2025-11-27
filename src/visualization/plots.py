import matplotlib.pyplot as plt
import pandas as pd

def plot_accuracy_by_position(df: pd.DataFrame, metric: str = "EM"):
    if df.empty or metric not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for {metric}", ha="center")
        return fig
    acc = df.groupby("position")[metric].mean().reindex(["start", "middle", "end"]).dropna()
    fig, ax = plt.subplots(figsize=(6,4))
    acc.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1 if metric in ["EM", "precision", "recall", "f1", "hit_rate", "mrr"] else None)
    ax.set_title(f"{metric} by answer position")
    ax.set_ylabel(metric)
    ax.set_xlabel("Position")
    for i, v in enumerate(acc.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    return fig

def plot_accuracy_by_context(df: pd.DataFrame, metric: str = "EM"):
    if df.empty or metric not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for {metric}", ha="center")
        return fig
    df = df.copy()
    df['ctx_bin'] = pd.qcut(df['context_tokens'].replace(0, 1), q=6, duplicates='drop')
    acc = df.groupby("ctx_bin", observed=False)[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(acc)), acc[metric], marker='o')
    ax.set_ylim(0, 1 if metric in ["EM", "precision", "recall", "f1", "hit_rate", "mrr"] else None)
    ax.set_title(f"{metric} vs. context length (binned)")
    ax.set_ylabel(metric)
    ax.set_xlabel("Context bins (increasing)")
    plt.tight_layout()
    return fig

def plot_heatmap(df: pd.DataFrame, metric: str = "EM"):
    """
    Plots a heatmap of metric vs. (Position, Context Length).
    X-axis: Context Length Bins
    Y-axis: Position (Start, Middle, End)
    """
    if df.empty or metric not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for {metric}", ha="center")
        return fig
    
    df = df.copy()
    # Bin context length
    # Use qcut for equal frequency or cut for equal width?
    # Given the user might test 1000, 2000, 3000... cut is probably better to see the scale.
    # But qcut is safer if data is sparse. Let's use qcut with duplicates drop.
    try:
        df['ctx_bin'] = pd.qcut(df['context_tokens'].replace(0, 1), q=5, duplicates='drop')
    except ValueError:
        # Fallback if too few unique values
        df['ctx_bin'] = df['context_tokens']

    # Pivot: Index=Position, Columns=Bin, Values=Metric
    pivot = df.groupby(["position", "ctx_bin"], observed=False)[metric].mean().unstack()
    
    # Reorder index if possible
    desired_order = ["start", "middle", "end"]
    pivot = pivot.reindex([p for p in desired_order if p in pivot.index])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot heatmap using imshow
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    
    # Axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    ax.set_title(f"{metric} Heatmap (Position vs Context)")
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Position")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
                
    plt.tight_layout()
    return fig
