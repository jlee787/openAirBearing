import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go

# Plot styling
PLOT_FONT = dict(
    family="Arial",
    size=12,
)

# Define solver colors at the top level with other constants
SOLVER_COLORS = {
    'Analytic': 'blue',
    'Numeric': 'red',
    'Numeric 2': 'green'
}

def plot_key_results(bearing, results):
    """Create four subplots for bearing visualization
    
    Args:
        bearing: Bearing instance with updated parameters
        results: List of (p, w, k, q) tuples from solve_axisymmetric_analytic
    """
    # Create figure with subplots
    fig = sp.make_subplots(
        rows=2, cols=3, 
        subplot_titles=('Load Capacity', 'Stiffness', 'Pressure Distribution',
                        'Supply Flow Rate', 'Chamber Flow Rate', 'Ambient Flow Rate', 
                        )
    )

     # Convert single inputs to lists for consistent handling
    results = [results] if not isinstance(results, list) else results

    k_min = 0

    for i, result in enumerate(results):
        color = SOLVER_COLORS.get(result.name, 'purple')
        
        k_max_idx = np.argmax(result.k)
        k_min = np.minimum(k_min, np.min(result.k))

        # Load capacity plot
        fig.add_trace(
            go.Scatter(
                x=bearing.ha.flatten() * 1e6, 
                y=result.w,
                name=result.name,
                mode='lines+markers',
                marker=dict(
                    color=color, 
                    size=[8 if i == k_max_idx else 0 for i in range(bearing.nr)], 
                    symbol='circle'),
                line=dict(color=color)
            ), 
            row=1, col=1
        )
        
        # Stiffness plot
        fig.add_trace(
            go.Scatter(
                x=bearing.ha.flatten() * 1e6,
                y=result.k * 1e-6,
                name=result.name,
                mode='lines+markers',
                marker=dict(
                    color=color, 
                    size=[8 if i == k_max_idx else 0 for i in range(bearing.nr)], 
                    symbol='circle'),
                line=dict(color=color),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Pressure distribution plot with distributed labels
        n_plots = 4
        h_plots = np.linspace(bearing.ha_min**0.5, bearing.ha_max**0.5, n_plots)**2
        t_locations = np.round(np.linspace(bearing.nr, 0, n_plots+2)[1:-1]).astype(int)
        
        for h_plot, t_loc in zip(h_plots, t_locations):
            in_h = np.abs(bearing.ha - h_plot).argmin()
            pressures = (result.p[:, in_h] - bearing.pa) * 1e-6  # Convert to MPa
            fig.add_trace(
                go.Scatter(
                    x=bearing.r * 1e3, y=pressures,
                    mode='lines+text',
                    textposition='top center',
                    text=[f"{h_plot*1e6:.2f} μm" if i == t_loc else None for i in range(bearing.nr)],
                    textfont=dict(color=color),
                    name=f"{result.name} {h_plot*1e6:.1f} μm", line=dict(color=color),
                    showlegend=False
                ),
                row=1, col=3
            )


        # Supply Flow rate plot
        fig.add_trace(
            go.Scatter(
                x=bearing.ha.flatten() * 1e6,
                y=result.qs,
                name=result.name,
                mode='lines+markers',
                marker=dict(
                    color=color, 
                    size=[8 if i == k_max_idx else 0 for i in range(bearing.nr)], 
                    symbol='circle'),
                line=dict(color=color),
                showlegend=False
            ),
            row=2, col=1
        )
        
        if bearing.type == "seal":
            # Chamber Flow rate plot
            fig.add_trace(
                go.Scatter(
                    x=bearing.ha.flatten() * 1e6,
                    y=result.qc,
                    name=result.name,
                    mode='lines+markers',
                    marker=dict(
                        color=color, 
                        size=[8 if i == k_max_idx else 0 for i in range(bearing.nr)], 
                        symbol='circle'),
                    line=dict(color=color),
                    showlegend=False
                ),
                row=2, col=2
            )

            # Ambient Flow rate plot
            fig.add_trace(
                go.Scatter(
                    x=bearing.ha.flatten() * 1e6,
                    y=result.qa,
                    name=result.name,
                    mode='lines+markers',
                    marker=dict(
                        color=color, 
                        size=[8 if i == k_max_idx else 0 for i in range(bearing.nr)], 
                        symbol='circle'),
                    line=dict(color=color),
                    showlegend=False
                ),
                row=2, col=3
            )
        else:
            fig.layout.annotations[4].text = "" # remove subplot title
            fig.layout.annotations[5].text = "" 
    

    # Common axis properties
    axis_style = dict(
        title_font=PLOT_FONT,
        tickfont=PLOT_FONT,
        showline=True,
        linecolor='black',
        ticks='inside',
        mirror=True
    )

    # Update axes
    for i in range(1, 3):
        for j in range(1, 4):
            fig.update_xaxes(axis_style, row=i, col=j)
            fig.update_yaxes(axis_style, row=i, col=j)

    # Update axis labels and ranges
    max_height = bearing.ha_max * 1e6
    max_radius = bearing.ra * 1e3

    fig.update_xaxes(title_text="h (μm)", range=[0, max_height], row=1, col=1)
    fig.update_xaxes(title_text="h (μm)", range=[0, max_height], row=1, col=2)
    fig.update_xaxes(title_text="h (μm)", range=[0, max_radius], row=1, col=3)

    fig.update_xaxes(title_text="h (μm)", range=[0, max_height], row=2, col=1)
    fig.update_xaxes(title_text="h (μm)", range=[0, max_height], row=2, col=2)
    fig.update_xaxes(title_text="r (mm)", range=[0, max_height], row=2, col=3)

    fig.update_yaxes(title_text="w (N)", range=[0, None], row=1, col=1)
    fig.update_yaxes(title_text="k (N/μm)", range=[k_min*1e-6, None], row=1, col=2)
    fig.update_yaxes(title_text="p (MPa)", range=[0, None], row=1, col=3)

    fig.update_yaxes(title_text="q<sub>s/sub> (l/min)", range=[0, None], row=2, col=1)
    #fig.update_yaxes(title_text="q<sub>c</sub> (l/min)", range=[None, 0], row=2, col=2)
    fig.update_yaxes(title_text="q<sub>a</sub> (l/min)", range=[0, None], row=2, col=3)

    # Update layout
    fig.update_layout(
        font=PLOT_FONT,
        height=500,
        #showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        )
    )

    return fig

def plot_bearing_shape(bearing):
    """Create bearing shape visualization
    
    Args:
        bearing: Bearing instance with geometry parameters
    """
    # Create figure
    fig = go.Figure()


    # fig.add_trace(
    #     go.Scatter(
    #         x=np.array([0, bearing.ra])* 1e3,  # Convert to mm
    #         y=np.array([1, 1])*1e6,
    #         showlegend=False
    #     )
    # )
    # # Add porous layer shape
    # fig.add_trace(
    #     go.Scatter(
    #         x=bearing.r * 1e3,  # Convert to mm
    #         y=np.zeros(bearing.nr), 
    #         fill='tonexty',
    #         fillcolor='rgba(200, 200, 255, 0.5)',
    #         line=dict(color='blue'),
    #         name='Analytic',
    #         showlegend=False
    #     )
    # )

    fig.add_trace(
        go.Scatter(
            x=np.array([bearing.r[1] * 1e3, bearing.r[-1] * 1e3]),  # Convert to mm
            y=np.array([100, 100]),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bearing.r * 1e3,  # Convert to mm
            y=bearing.geom * 1e6,  # Convert to um
            fill='tonexty',
            # fillcolor='rgba(255, 200, 200, 0.5)',
            # line=dict(color='red'),
            fillcolor='lightgrey',
            line=dict(color='black'),
            name='Shape',
            showlegend=False
        )
    )
    
    # Update layout
    fig.update_layout(
        font=PLOT_FONT,
        height=200,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title="r (mm)",
            showline=True,
            linecolor='black',
            mirror=True,
            range=[0, bearing.ra * 1e3]
        ),
        yaxis=dict(
            title="Shape (μm)",
            showline=True,
            linecolor='black',
            mirror=True,
            range=[min(bearing.geom) * 1.2 * 1e6, max(bearing.geom) * 1.2 * 1e6]
        )
    )
    return fig
