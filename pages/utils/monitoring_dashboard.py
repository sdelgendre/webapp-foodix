import matplotlib.pyplot as plt

def get_conditional_formatting_color(metric, value):
    if metric == "temperature":
        if value > 9:
            return "red"
        if value > 7:
            return "orange"
        else:
            return "green"
    if metric == "humidity":
        if value > 0.55:
            return "red"
        if value > 0.5:
            return "orange"
        else:
            return "green"
    if metric == "pressure":
        if value < 990:
            return "red"
        if value < 1_000:
            return "orange"
        else:
            return "green"


def plot_monitoring_dashboard(silosDataFrame, siloIndex, fillValues):
    fig, (a1, a2, a3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 2, 3]})

    x_positions = [0.2, 0.5, 0.8]
    metrics = ["temperature", "humidity", "pressure"]
    metrics_name = ["Temperature (°C)", "Humidity (%)", "Pressure (hPa)"]

    a1.axis('off')
    a1.text(
        x=0.5,
        y=0.5,
        s=silosDataFrame["name"].iloc[siloIndex],
        fontsize=30,
        ha="center"
    )

    a2.axis("off")
    for i in range(3):
        a2.text(
            x=x_positions[i], 
            y=0.7, 
            s=silosDataFrame[metrics[i]].iloc[siloIndex], 
            fontsize=15,
            color=get_conditional_formatting_color(metrics[i], silosDataFrame[metrics[i]].iloc[siloIndex]),
            ha="center"
        )
        a2.text(
            x=x_positions[i], 
            y=0.4, 
            s=metrics_name[i], 
            fontsize=10,
            color=get_conditional_formatting_color(metrics[i], silosDataFrame[metrics[i]].iloc[siloIndex]),
            ha="center"
        )

    a3.plot(
        fillValues["dates"], 
        fillValues[silosDataFrame["name"].iloc[siloIndex]]
    )
    ticks       = a3.get_xticks()[::10]
    ticklabels  = a3.get_xticklabels()[::10]
    a3.set_xticks(ticks=ticks)
    a3.set_xticklabels(
        labels=ticklabels, 
        rotation=20, 
        fontsize=6
    )
    a3.set_yticklabels(
        labels=a3.get_yticklabels(), 
        fontsize=8
    )
    a3.set_ylabel("Silo filling level", fontsize=10)
    a3.spines.right.set_visible(False)
    a3.spines.top.set_visible(False)
    
    return fig
