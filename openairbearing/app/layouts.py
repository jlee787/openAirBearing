from dash import html, dcc

from openairbearing.plots import plot_bearing_shape, plot_key_results
from openairbearing.config import DEMO_MODE


STYLES = {
    "input": {
        "width": "100px",
        "borderRadius": "4px",
        "border": "1px solid #ddd",
        "padding": "4px",
    },
    "input_container": {
        "display": "grid",
        "grid-template-columns": "200px 100px 20px",
        "marginBottom": "20px",
        "gap": "20px",
        "align-items": "center",
    },
    "toggle_container": {
        "display": "none",
        "grid-template-columns": "200px 100px 20px",
        "marginTop": "20px",
        "gap": "20px",
        "align-items": "center",
    },
    "reset_button": {
        "padding": "2px 6px",
        "fontSize": "14px",
        "backgroundColor": "#f8f9fa",
        "border": "1px solid #ddd",
        "borderRadius": "4px",
        "cursor": "pointer",
        "height": "25px",
        "width": "30px",
    },
    "header_container": {
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "marginBottom": "20px",
    },
    "reset_all_button": {
        "padding": "5px 10px",
        "fontSize": "14px",
        "backgroundColor": "#f8f9fa",
        "border": "1px solid #ddd",
        "borderRadius": "4px",
        "cursor": "pointer",
        "height": "30px",
    },
    "input_column": {
        "width": "370px",
        "minWidth": "370px",  # Prevent inputs from getting too narrow
        "display": "inline-block",
        "vertical-align": "top",
        "padding": "20px",
        "border": "1px solid black",
        "borderRadius": "8px",
        "flex": "0 1 auto",  # Don't grow, allow shrink, auto basis
    },
    "plot_box": {
        "width": "calc(100% - 40px)",  # Dynamic width based on container
        "display": "inline-block",
        "vertical-align": "top",
        "justifyContent": "space-between",
        "padding": "20px",
        "border": "1px solid black",
        "borderRadius": "8px",
        "flex": "1 1 auto",  # Allow grow and shrink
    },
    "plot_column": {
        "width": "calc(100% - 550px)",  # Dynamic width based on container
        "minWidth": "600px",  # Minimum width for plots
        "display": "inline-block",
        "vertical-align": "top",
        "padding": "0px",
        "flex": "1 1 auto",  # Allow grow and shrink
    },
    "container": {
        "display": "flex",
        "alignItems": "flex-start",
        "justifyContent": "space-between",
        "width": "100%",
        "flexWrap": "wrap",  # Allow wrapping on smaller screens
        "gap": "20px",  # Space between columns
    },
}


def create_layout(default_bearing, bearing, results):
    """Create the main app layout.

    Args:
        bearing: Bearing instance
        results: List of calculation results

    Returns:
        html.Div: Main application layout
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Img(
                        src="/assets/favicon.ico",
                        style={
                            "height": "40px",
                            "margin": "10px 5px",
                            "verticalAlign": "middle",
                        },
                    ),
                    html.H1(
                        "Open Air Bearing",
                        style={
                            "textAlign": "center",
                            "display": "inline-block",
                            "margin": "10px 0",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            ),
            html.Div(
                [
                    create_input_layout(default_bearing),
                    create_results_layout(bearing, results),
                ],
                style=STYLES["container"],
            ),
        ]
    )


def create_input_layout(default_bearing):
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Setup", style={"margin": "0"}),
                    html.Button(
                        "Reset All",
                        id="reset-all",
                        title="Reset all values to default",
                        style=STYLES["reset_all_button"],
                    ),
                ],
                style=STYLES["header_container"],
            ),
            # html.H4("Setup"),
            html.Div(
                [
                    html.Label("Simulated Case"),
                    dcc.Dropdown(
                        id="case-select",
                        options=[
                            {"label": "Circular thrust", "value": "circular"},
                            {"label": "Annular thrust", "value": "annular"},
                            {
                                "label": "Infinitely long",
                                "value": "infinite",
                                "disabled": False,
                            },
                            {
                                "label": "Rectangular",
                                "value": "rectangular",
                                "disabled": False,
                            },
                            {"label": "Journal", "value": "journal", "disabled": False},
                        ],
                        value="circular",
                        style={"width": "150px"},
                    ),
                    html.Label(""),
                    html.Label("Solution selection"),  # Fixed capitalization here
                    dcc.Checklist(
                        id="solver-select",
                        options=[
                            {
                                "label": "Analytic",
                                "value": "analytic",
                                "disabled": False,
                            },
                            {
                                "label": "Numeric",
                                "value": "numeric",
                                "disabled": False,
                            },
                            {
                                "label": "Numeric 2d",
                                "value": "numeric2d",
                                "disabled": False,
                            },
                        ],
                        value=["analytic"],
                        style={"color": "black", "width": "150px"},
                    ),
                ],
                style=STYLES["input_container"],
            ),
            # Geometry inputs
            html.H4("Bearing parameters"),
            html.Div(
                [
                    html.Label("Porous Layer Thickness (mm)"),
                    dcc.Input(
                        id="hp-input",
                        type="number",
                        min=0.01,
                        value=default_bearing.hp * 1e3,  # Convert m to mm
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="hp-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                    html.Label("Outer radius / length (mm)"),
                    dcc.Input(
                        id="xa-input",
                        type="number",
                        min=0.01,
                        value=default_bearing.xa * 1e3,  # Convert m to mm
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="xa-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                ],
                style=STYLES["input_container"],
            ),
            html.Div(
                [
                    html.Label("Inner radius (mm)"),
                    dcc.Input(
                        id="xc-input",
                        type="number",
                        min=0.01,
                        value=default_bearing.xc * 1e3,  # Convert m to mm
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="xc-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                ],
                id="xc-container",
                style=STYLES["toggle_container"],
            ),
            html.Div(
                [
                    html.Label("Length (mm)"),
                    dcc.Input(
                        id="ya-input",
                        type="number",
                        min=0.01,
                        value=default_bearing.ya * 1e3,  # Convert m to mm
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="ya-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                ],
                id="ya-container",
                style=STYLES["toggle_container"],
            ),
            html.Div(
                [
                    html.Label("Permeability (m^2)"),
                    dcc.Input(
                        id="kappa-input",
                        type="number",
                        value=default_bearing.kappa,
                        min=0,
                        step=1e-16,
                        inputMode="numeric",
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="kappa-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                    html.Label("Free flow (l/min)"),
                    dcc.Input(
                        id="Qsc-input",
                        type="number",
                        value=default_bearing.Qsc,
                        min=0.1,
                        step=0.1,
                        inputMode="numeric",
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="Qsc-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                ],
                style=STYLES["input_container"],
            ),
            # Geometry inputs
            html.H4("Numerical model specific:"),
            html.Div(
                [
                    html.Label("Geometrical error type"),
                    dcc.Dropdown(
                        id="error-select",
                        options=[
                            {
                                "label": "Linear",
                                "value": "linear",
                            },
                            {
                                "label": "Quadratic",
                                "value": "quadratic",
                            },
                            {
                                "label": "Tilt x",
                                "value": "tiltx",
                            },
                            {
                                "label": "Tilt y",
                                "value": "tilty",
                            },
                        ],
                        value="linear",
                        style={"width": "150px"},
                    ),
                    html.Label(""),
                    html.Label("Geometry error (μm)"),
                    dcc.Input(
                        id="error-input",
                        type="number",
                        step=0.5,
                        value=default_bearing.error * 1e6,
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="error-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                    html.Label("Slip coefficeint Φ"),
                    dcc.Input(
                        id="psi-input",
                        type="number",
                        min=0,
                        step=0.01,
                        value=default_bearing.Psi * 1e6,
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="psi-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                ],
                style=STYLES["input_container"],
            ),
            html.H4("Load parameters"),
            html.Div(
                [
                    html.Label("Ambient Pressure (MPa)"),
                    dcc.Input(
                        id="pa-input",
                        type="number",
                        value=default_bearing.pa * 1e-6,  # Convert Pa to MPa
                        min=0,
                        step=0.1,
                        inputMode="numeric",
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="pa-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                    html.Label("Supply Pressure (MPa)"),
                    dcc.Input(
                        id="ps-input",
                        type="number",
                        value=default_bearing.ps * 1e-6,  # Convert Pa to MPa
                        min=0.1,
                        step=0.1,
                        inputMode="numeric",
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="ps-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                ],
                style=STYLES["input_container"],
            ),
            html.Div(
                [
                    html.Label("Chamber Pressure (MPa)"),
                    dcc.Input(
                        id="pc-input",
                        type="number",
                        value=default_bearing.pc * 1e-6,  # Convert Pa to MPa
                        min=0.1,
                        step=0.1,
                        inputMode="numeric",
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="pc-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                ],
                id="pc-container",
                style=STYLES["toggle_container"],
            ),
            html.H4("Fluid properties"),
            html.Div(
                [
                    html.Label("Air Density (kg/m³)"),
                    dcc.Input(
                        id="rho-input",
                        type="number",
                        value=default_bearing.rho,
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="rho-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                    html.Label("Dynamic Viscosity (Pa·s)"),
                    dcc.Input(
                        id="mu-input",
                        type="number",
                        value=default_bearing.mu,
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="mu-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                ],
                style=STYLES["input_container"],
            ),
            html.H4("Model parameters"),
            html.Div(
                [
                    html.Label("Minimum Height (μm)"),
                    dcc.Input(
                        id="ha-min-input",
                        type="number",
                        value=default_bearing.ha_min * 1e6,  # Convert m to μm
                        min=0,
                        step=0.5,
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="ha-min-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                    html.Label("Maximum Height (μm)"),
                    dcc.Input(
                        id="ha-max-input",
                        type="number",
                        value=default_bearing.ha_max * 1e6,  # Convert m to μm
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="ha-max-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                    html.Label("Number of height points"),
                    dcc.Input(
                        id="nh-input",
                        type="number",
                        min=3,
                        max=100 if DEMO_MODE else None,
                        step=1,
                        value=default_bearing.nh,
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="nh-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                    html.Label("Number of x direction points"),
                    dcc.Input(
                        id="nx-input",
                        type="number",
                        value=default_bearing.nx,
                        min=3,
                        max=100 if DEMO_MODE else None,
                        step=1,
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="nx-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                ],
                style=STYLES["input_container"],
            ),
            html.Div(
                [
                    html.Label("Number of y direction points"),
                    dcc.Input(
                        id="ny-input",
                        type="number",
                        value=default_bearing.ny,
                        min=3,
                        max=100 if DEMO_MODE else None,
                        step=1,
                        style=STYLES["input"],
                    ),
                    html.Button(
                        "↺",
                        id="ny-reset",
                        title="Reset to default",
                        style=STYLES["reset_button"],
                    ),
                ],
                id="ny-container",
                style=STYLES["toggle_container"],
            ),
        ],
        style=STYLES["input_column"],
    )


def create_results_layout(bearing, results):
    """Create the results section layout.

    Args:
        bearing: Bearing instance
        results: List of calculation results
    """
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Bearing shape", style={"margin": "0"}),
                    dcc.Graph(
                        id="bearing-shape",
                        figure=plot_bearing_shape(bearing),
                        config={"displayModeBar": False},
                    ),
                ],
                style=STYLES["plot_box"],
            ),
            # Spacer
            html.Div(style={"height": "20px"}),
            html.Div(
                [
                    html.H3("Results", style={"margin": "0"}),
                    dcc.Graph(
                        id="bearing-plots",
                        figure=plot_key_results(bearing, results),
                        config={"displayModeBar": False},
                    ),
                ],
                style=STYLES["plot_box"],
            ),
        ],
        style=STYLES["plot_column"],
    )
