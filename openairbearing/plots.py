import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Plot styling
PLOT_FONT = dict(
    family="Arial",
    size=12,
)

# Define solver colors at the top level with other constants
SOLVER_COLORS = {
    "analytic": "blue",
    "numeric": "red",
    "numeric2d": "red",
}

# Common axis properties
AXIS_STYLE = dict(
    title_font=PLOT_FONT,
    tickfont=PLOT_FONT,
    showline=True,
    linecolor="black",
    ticks="inside",
    mirror=True,
)


def plot_key_results(bearing, results):
    """Create four subplots for bearing visualization

    Args:
        bearing: Bearing instance with updated parameters
        results: List of (p, w, k, q) tuples from solve_bearing
    """
    results = [results] if not isinstance(results, list) else results
    if not results:
        fig = go.Figure()
        fig.update_layout(
            title="No suitable solver selected",
            font=dict(family="Arial", size=12),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    b = bearing

    # Create figure with subplots
    fig = sp.make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Load Capacity",
            "Stiffness",
            "Pressure Distribution",
            "Supply Flow Rate",
            "Chamber Flow Rate",
            "Ambient Flow Rate",
        ),
    )

    # Update axes
    for i in range(1, 3):
        for j in range(1, 4):
            fig.update_xaxes(AXIS_STYLE, row=i, col=j)
            fig.update_yaxes(AXIS_STYLE, row=i, col=j)

    # Update axis labels and ranges
    max_height = b.ha_max * 1e6
    max_radius = b.xa * 1e3
    k_min = 0

    for i, result in enumerate(results):
        color = SOLVER_COLORS.get(result.name, "purple")

        k_max_idx = np.argmax(result.k)
        k_min = np.minimum(k_min, np.min(result.k))

        # Load capacity plot
        fig.add_trace(
            go.Scatter(
                x=b.ha.flatten() * 1e6,
                y=result.w,
                name=result.name,
                mode="lines+markers",
                marker=dict(
                    color=color,
                    size=[8 if i == k_max_idx else 0 for i in range(b.nx)],
                    symbol="circle",
                ),
                line=dict(color=color),
            ),
            row=1,
            col=1,
        )

        # Stiffness plot
        fig.add_trace(
            go.Scatter(
                x=b.ha.flatten() * 1e6,
                y=result.k * 1e-6,
                name=result.name,
                mode="lines+markers",
                marker=dict(
                    color=color,
                    size=[8 if i == k_max_idx else 0 for i in range(b.nx)],
                    symbol="circle",
                ),
                line=dict(color=color),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        n_plots = 4
        h_plots = np.linspace(b.ha_min**0.5, b.ha_max**0.5, n_plots) ** 2
        t_locations = np.round(np.linspace(b.nx, 0, n_plots + 2)[1:-1]).astype(int)
        if b.ny == 1 or result.name == "analytic":
            for h_plot, t_loc in zip(h_plots, t_locations):
                in_h = np.abs(b.ha - h_plot).argmin()
                pressures = (result.p[:, in_h] - b.pa) * 1e-6  # Convert to MPa
                fig.add_trace(
                    go.Scatter(
                        x=b.x * 1e3,
                        y=pressures,
                        mode="lines+text",
                        textposition="top center",
                        text=[
                            f"{h_plot*1e6:.2f} μm" if i == t_loc else None
                            for i in range(b.nx)
                        ],
                        textfont=dict(color=color),
                        name=f"{result.name} {h_plot*1e6:.1f} μm",
                        line=dict(color=color),
                        showlegend=False,
                    ),
                    row=1,
                    col=3,
                )
            fig.update_xaxes(title_text="r (mm)", range=[0, max_radius], row=1, col=3)
            fig.update_yaxes(title_text="p (MPa)", range=[0, None], row=1, col=3)
        else:
            # for h_plot, t_loc in zip(h_plots, t_locations):
            #     in_h = np.abs(b.ha - h_plot).argmin()
            #     pressures = (result.p[3, :, in_h] - b.pa) * 1e-6  # Convert to MPa
            #     fig.add_trace(
            #         go.Scatter(
            #             x=b.x * 1e3,
            #             y=pressures,
            #             mode="lines+text",
            #             textposition="top center",
            #             text=[
            #                 f"{h_plot*1e6:.2f} μm" if i == t_loc else None
            #                 for i in range(b.nx)
            #             ],
            #             textfont=dict(color=color),
            #             name=f"{result.name} {h_plot*1e6:.1f} μm",
            #             line=dict(color=color),
            #             showlegend=False,
            #         ),
            #         row=1,
            #         col=3,
            #     )
            pressures = (result.p - b.pa) * 1e-6  # Convert to MPa
            fig.add_trace(
                go.Contour(
                    z=pressures[:, :, k_max_idx],
                    x=b.x * 1e3,
                    y=b.y * 1e3,
                    colorscale="Viridis",
                    zmin=0,
                    zmax=np.max(pressures),
                    contours=dict(
                        coloring="heatmap",
                        showlabels=True,
                        labelfont=dict(size=10, color="white"),
                    ),
                    colorbar=dict(
                        title="(MPa)",
                        x=1.01,
                        y=0.61,
                        xanchor="left",
                        yanchor="bottom",
                        len=0.45,
                        thickness=15,
                    ),
                    name="Air gap pressure",
                ),
                row=1,
                col=3,
            )

            fig.update_xaxes(
                title_text="x (mm)",
                range=[b.x.min() * 1e3, b.x.max() * 1e3],
                row=1,
                col=3,
            )

            fig.update_yaxes(
                title_text="y (mm)",
                range=[b.y.min() * 1e3, b.y.max() * 1e3],
                row=1,
                col=3,
            )

            # slider for h selection
            steps = []
            for i, h in enumerate(bearing.ha * 1e6):
                step = dict(
                    method="update",
                    args=[
                        {"z": [pressures[:, :, i]]},
                        {
                            "title": f"Pressure Distribution (h = {h:.1f})"
                        },  # Update the title
                    ],
                    label=f"{h:.1f} μm",
                )
                steps.append(step)

            sliders = [
                dict(
                    active=k_max_idx,
                    currentvalue={
                        "prefix": "Pressure distribution plot height: ",
                        "font": {"size": 14},
                    },
                    pad={"t": 50},
                    steps=steps,
                )
            ]

            fig.update_layout(sliders=sliders)

        # Supply Flow rate plot
        fig.add_trace(
            go.Scatter(
                x=b.ha.flatten() * 1e6,
                y=result.qs,
                name=result.name,
                mode="lines+markers",
                marker=dict(
                    color=color,
                    size=[8 if i == k_max_idx else 0 for i in range(b.nx)],
                    symbol="circle",
                ),
                line=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        if b.type == "seal":
            try:
                # Chamber Flow rate plot
                fig.add_trace(
                    go.Scatter(
                        x=b.ha.flatten() * 1e6,
                        y=result.qc,
                        name=result.name,
                        mode="lines+markers",
                        marker=dict(
                            color=color,
                            size=[8 if i == k_max_idx else 0 for i in range(b.nx)],
                            symbol="circle",
                        ),
                        line=dict(color=color),
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )

                # Ambient Flow rate plot
                fig.add_trace(
                    go.Scatter(
                        x=b.ha.flatten() * 1e6,
                        y=result.qa,
                        name=result.name,
                        mode="lines+markers",
                        marker=dict(
                            color=color,
                            size=[8 if i == k_max_idx else 0 for i in range(b.nx)],
                            symbol="circle",
                        ),
                        line=dict(color=color),
                        showlegend=False,
                    ),
                    row=2,
                    col=3,
                )
            except:
                print(
                    "Warning: Chamber and Ambient Flow rates not available for this bearing type."
                )
        else:
            fig.layout.annotations[4].text = (
                ""  # remove subplot titles for missing plots
            )
            fig.layout.annotations[5].text = ""

    fig.update_xaxes(title_text="h (μm)", range=[0, max_height], row=1, col=1)
    fig.update_xaxes(title_text="h (μm)", range=[0, max_height], row=1, col=2)
    # r1 c3 above

    fig.update_xaxes(title_text="h (μm)", range=[0, max_height], row=2, col=1)
    fig.update_xaxes(title_text="h (μm)", range=[0, max_height], row=2, col=2)
    fig.update_xaxes(title_text="h (μm)", range=[0, max_height], row=2, col=3)

    fig.update_yaxes(title_text="w (N)", range=[0, None], row=1, col=1)
    fig.update_yaxes(title_text="k (N/μm)", range=[k_min * 1e-6, None], row=1, col=2)
    # r1 c3 above

    fig.update_yaxes(title_text="q<sub>s</sub> (l/min)", range=[0, None], row=2, col=1)
    fig.update_yaxes(
        title_text="q<sub>c</sub> (l/min)", range=[None, None], row=2, col=2
    )
    fig.update_yaxes(
        title_text="q<sub>a</sub> (l/min)", range=[None, None], row=2, col=3
    )

    # Update layout
    fig.update_layout(
        font=PLOT_FONT,
        height=800,
        # showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
    )

    return fig


def plot_bearing_shape(bearing):
    """Create bearing shape visualization

    Args:
        bearing: Bearing instance with geometry parameters
    """
    b = bearing
    # Create figure
    # fig = go.Figure()
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("XY Geometry", "XZ Profile"))

    plot_xz_shape(b, fig)
    plot_xy_shape(b, fig)

    for i in range(1, 3):
        fig.update_xaxes(AXIS_STYLE, row=1, col=i)
        fig.update_yaxes(AXIS_STYLE, row=1, col=i)

    fig.update_layout(
        font=PLOT_FONT,
        height=300,
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
    )
    return fig


def plot_xy_shape(bearing, fig):
    b = bearing
    # SHAPE XY
    match b.case:
        case "annular" | "circular":

            # outer circle
            theta = np.linspace(0, 2 * np.pi, 100)
            xa = b.xa * np.cos(theta) * 1e3
            ya = b.xa * np.sin(theta) * 1e3
            fig.add_trace(
                go.Scatter(
                    x=xa,
                    y=ya,
                    fill="toself",
                    fillcolor="lightgrey",
                    line=dict(color="black"),
                    name="Shape",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # inner hole
            if b.case == "annular":
                xc = b.xc * np.cos(theta) * 1e3
                yc = b.xc * np.sin(theta) * 1e3
                fig.add_trace(
                    go.Scatter(
                        x=xc,
                        y=yc,
                        fill="toself",
                        fillcolor="white",
                        line=dict(color="black"),
                        name="Shape",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

            # symmetry lines
            fig.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[-b.xa * 0.2e3, b.xa * 0.2e3],
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dashdot"),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[-b.xa * 0.2e3, b.xa * 0.2e3],
                    y=[0, 0],
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dashdot"),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        case "infinite":
            # right edge
            fig.add_trace(
                go.Scatter(
                    x=np.array([1, 1]) * b.xa * 1e3,
                    y=np.array([0, 1000]),
                    mode="lines",
                    line=dict(color="black"),
                    name="Shape",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            # left edge
            fig.add_trace(
                go.Scatter(
                    x=np.array([0, 0]),
                    y=np.array([0, 1000]),
                    fill="tonextx",
                    fillcolor="lightgrey",
                    mode="lines",
                    line=dict(color="black"),
                    name="Shape",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            fig.update_yaxes(range=[0, 1000], row=1, col=1)

            fig.update_xaxes(
                range=np.array([-0.5, 1.5]) * b.xa * 1e3,
            )

        case "rectangular":
            fig.add_trace(
                go.Scatter(
                    x=np.array([-1, -1, 1, 1, -1]) * b.xa * 0.5e3,
                    y=np.array([-1, 1, 1, -1, -1]) * b.ya * 0.5e3,
                    fill="toself",
                    fillcolor="lightgrey",
                    mode="lines",
                    line=dict(color="black"),
                    name="Shape",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.update_yaxes(
                range=np.array([-0.6, 0.6]) * b.ya * 1e3,
            )
            fig.update_xaxes(
                range=np.array([-0.6, 0.6]) * b.xa * 1e3,
                scaleanchor="y",
                scaleratio=1,
            )

        case _:
            pass

    fig.update_xaxes(
        title="x (mm)", showline=True, linecolor="black", mirror=True, row=1, col=1
    )
    fig.update_yaxes(
        title="y (mm)",
        showline=True,
        linecolor="black",
        mirror=True,
        scaleanchor=False if b.csys == "cartesian" else "x",
        scaleratio=1,
        row=1,
        col=1,
    )
    return fig


def plot_xz_shape(bearing, fig):
    b = bearing

    if b.ny > 1:
        if b.case == "circular" or b.case == "annular":
            zr = (b.geom.T).flatten()
            xr = (b.x[None, :] * np.cos(b.y)[:, None]).flatten()
            yr = (b.x[None, :] * np.sin(b.y)[:, None]).flatten()

            print(xr.shape, yr.shape, zr.shape)
            # Define a regular grid over the data
            x = np.linspace(-b.xa, b.xa, 99)
            y = x
            xg, yg = np.meshgrid(x, y)
            # evaluate the z-values at the regular grid through cubic interpolation
            z = griddata((xr, yr), zr, (xg, yg), method="cubic")
            if b.case == "annular":
                dist = np.sqrt(xg**2 + yg**2)
                z[dist < b.xc] = np.nan
            fig.update_xaxes(
                title_text="x (mm)",
                range=np.array([-1.1, 1.1]) * b.xa * 1e3,
                scaleanchor="y",
                row=1,
                col=2,
            )

            fig.update_yaxes(
                title_text="y (mm)",
                range=np.array([-1.1, 1.1]) * b.xa * 1e3,
                row=1,
                col=2,
            )

        elif b.case == "rectangular":
            z = b.geom.T
            x = b.x
            y = b.y
            fig.update_xaxes(
                title_text="x (mm)",
                range=np.array([-0.6, 0.6]) * b.xa * 1e3,
                row=1,
                col=2,
            )
            fig.update_yaxes(
                title_text="y (mm)",
                range=np.array([-0.6, 0.6]) * b.ya * 1e3,
                row=1,
                col=2,
            )
        elif b.case == "journal":
            z = b.geom.T
            x = b.x / b.xa
            y = b.y
            fig.update_xaxes(
                title_text="theta (rad)",
                range=np.array([-1.1, 1.1]) * np.pi,
                row=1,
                col=2,
            )
            fig.update_yaxes(
                title_text="y (mm)",
                range=np.array([-0.6, 0.6]) * b.ya * 1e3,
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Contour(
                z=z * 1e6,
                x=x * 1e3,
                y=y * 1e3,
                colorscale="Viridis",
                zmin=0,
                zmax=b.error * 1e6,
                contours=dict(
                    coloring="heatmap",
                    showlabels=True,
                    labelfont=dict(size=10, color="white"),
                ),
                colorbar=dict(title="(μm)", thickness=15),
                name="Profile",
            ),
            row=1,
            col=2,
        )

    else:
        # PROFILE XZ
        match b.case:
            case "circular":
                x = np.concatenate(([-b.x[-1]], -np.flip(b.x), b.x, [b.x[-1]])) * 1e3
                y = np.concatenate(([100], np.flip(b.geom), b.geom, [100])) * 1e6
                t = ["Bearing<br>" if i == b.nx else None for i in range(b.nx * 2)]
            case "annular":
                x = (
                    np.concatenate(
                        ([-b.x[-1]], -np.flip(b.x), [b.x[1]], [b.x[1]], b.x, [b.x[-1]])
                    )
                    * 1e3
                )
                y = (
                    np.concatenate(
                        ([100], np.flip(b.geom), [100], [100], b.geom, [100])
                    )
                    * 1e6
                )
                t = [
                    (
                        "Bearing<br>"
                        if i == i in [b.nx // 2, 2 + b.nx + b.nx // 2]
                        else None
                    )
                    for i in range(b.nx * 2)
                ]
            case "infinite":
                x = np.concatenate(([b.x[1]], b.x, [b.x[-1]])) * 1e3
                y = np.concatenate(([100], b.geom, [100])) * 1e6
                t = ["Bearing<br>" if i == b.nx // 2 else None for i in range(b.nx)]
            case _:
                pass

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                fillcolor="lightgrey",
                mode="lines+text",
                textposition="top center",
                text=t,
                textfont=dict(size=14),
                line=dict(color="black"),
                name="Bearing",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

        # Guide surface XZ
        fig.add_trace(
            go.Scatter(
                x=np.array(
                    [
                        (x[-1] - x[-1] * 1.2 if x[1] == 0 else x[1] * 1.2),
                        (x[1] + x[-1]) / 2,
                        x[-1] * 1.2,
                    ]
                ),  # Convert to mm
                y=np.ones(3) * -0.1,  # Convert to um
                mode="lines+text",
                textposition="bottom center",
                text=[None, "<br>Guide surface", None],
                textfont=dict(size=14),
                line=dict(color="gray"),
                name="Shape",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # symmetry line
        if b.csys == "polar":
            fig.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[-100, 100],
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dashdot"),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        fig.update_xaxes(
            title="x (mm)",
            range=[(x[-1] - x[-1] * 1.1 if x[1] == 0 else x[1] * 1.1), x[-1] * 1.1],
            row=1,
            col=2,
        )
        fig.update_yaxes(
            title="Shape (μm)",
            range=[-0.5 - 0.3e6 * abs(b.error), 1 + np.max(b.geom) * 1e6],
            row=1,
            col=2,
        )
    return fig
