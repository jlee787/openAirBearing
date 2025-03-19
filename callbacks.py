from dash.dependencies import Input, Output
import dash

from plots import plot_bearing_shape, plot_key_results
from airbearings import *

def register_callbacks(app):
    """Register all callbacks for the application."""
    
    def get_default_bearing(case):
        """Return default bearing instance based on case."""
        match case:
                case 'Circular thrust':
                    return CircularBearing()
                case 'Annular thrust':
                    return AnnularBearing()
                case 'Infinitely long':
                    return InfiniteLinearBearing()
                                         
    # Add callback to handle parameter updates
    @app.callback(
        [
            Output('bearing-plots', 'figure'),
            Output('bearing-shape', 'figure'),
            Output('kappa-input', 'value', allow_duplicate=True),
            Output('Qsc-input', 'value', allow_duplicate=True),
        ],
        [
            Input('case-select', 'value'),
            Input('solver-select', 'value'),
            Input('pa-input', 'value'),
            Input('ps-input', 'value'),
            Input('pc-input', 'value'),
            Input('rho-input', 'value'),
            Input('mu-input', 'value'),
            Input('hp-input', 'value'),
            Input('ra-input', 'value'),
            Input('nr-input', 'value'),
            Input('ha-min-input', 'value'),
            Input('ha-max-input', 'value'),
            Input('ha-n-input', 'value'),
            Input('kappa-input', 'value'),
            Input('Qsc-input', 'value'),
            Input('c-input', 'value'),
            Input('psi-input', 'value')

        ],
        prevent_initial_call='initial_duplicate'
    )

    def update_bearing(case, solvers, pa_mpa, ps_mpa, pc_mpa, rho, mu, hp_mm, ra_mm, nr, 
                  ha_min_um, ha_max_um, n_ha, kappa, Qsc, c_um, Psi):
        """Update bearing parameters and recalculate results.
    
        Args:
            case: Selected bearing case
            solvers: List of selected solvers
            pa_mpa: Ambient pressure in MPa
            ...
        
        Returns:
            tuple: Updated figures and values
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
            
        input_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        try:
            # Get default bearing for selected case
            match case:
                case 'Circular thrust':
                    bearing_class = CircularBearing
                case 'Annular thrust':
                    bearing_class = AnnularBearing
                case 'Infinitely long':
                    bearing_class = InfiniteLinearBearing
                case _:
                    raise ValueError(f"Invalid case: {case}")
                
            # Create new bearing instance with appropriate class
            bearing = bearing_class(
                pa=pa_mpa * 1e6,
                ps=ps_mpa * 1e6,
                pc=pc_mpa * 1e6,
                rho=rho,
                mu=mu,
                hp=hp_mm * 1e-3,
                ra=ra_mm * 1e-3,
                nr=int(nr),
                ha_min=ha_min_um * 1e-6,
                ha_max=ha_max_um * 1e-6,
                n_ha=int(n_ha),
                c=c_um * 1e-6,
                Psi=Psi
            )

            # Update kappa or Qsc based on which input changed
            if input_id == 'kappa-input' and kappa is not None:
                bearing.kappa = kappa
                bearing.Qsc = get_Qsc(bearing)
                new_kappa = kappa
                new_Qsc = bearing.Qsc
            elif input_id == 'Qsc-input' and Qsc is not None:
                bearing.Qsc = Qsc
                bearing.kappa = get_kappa(bearing)
                new_kappa = bearing.kappa
                new_Qsc = Qsc
            else:
                # For other inputs, maintain current values
                bearing.kappa = kappa if kappa is not None else get_kappa(bearing)
                bearing.Qsc = Qsc if Qsc is not None else get_Qsc(bearing)
                new_kappa = bearing.kappa
                new_Qsc = bearing.Qsc

            bearing.beta = get_beta(bearing)
            
            # Calculate results for each selected solver
            results = []

            if 'analytic' in solvers:
                results.append(solve_bearing(bearing, soltype=ANALYTIC))
            if 'numeric' in solvers:
                results.append(solve_bearing(bearing, soltype=NUMERIC))

            return (plot_key_results(bearing, results),
                    plot_bearing_shape(bearing),
                    new_kappa, new_Qsc)
        except Exception as e:
            print(f"Error in calculation: {e}")
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    @app.callback(
        Output('chamber-pressure-container', 'style'),
        Input('case-select', 'value')
    )

    def toggle_chamber_pressure(case):
        """Show/hide chamber pressure input based on selected case."""
        base_style = {
            'grid-template-columns': '200px 100px 20px',
            'gap': '20px',
            'marginTop': '20px',  # Fixed typo and using camelCase
            'marginBottom': '20px',
            'align-items': 'center'
        }
    
        if case == 'Annular thrust' or case == 'Infinitely long':
            return {**base_style, 'display': 'grid'}
        return {**base_style, 'display': 'none'}

    @app.callback(
        [
            Output('rho-input', 'value'),
            Output('mu-input', 'value'),
            Output('hp-input', 'value'),
            Output('ra-input', 'value'),
            Output('kappa-input', 'value', allow_duplicate=True),
            Output('Qsc-input', 'value', allow_duplicate=True),
            Output('pa-input', 'value'),
            Output('pc-input', 'value'),
            Output('ps-input', 'value'),
            Output('ha-min-input', 'value'),
            Output('ha-max-input', 'value'),
            Output('nr-input', 'value'),
            Output('ha-n-input', 'value'),
            Output('c-input', 'value'),
            Output('psi-input', 'value')
         ],
        [
            Input('reset-all', 'n_clicks'),
            Input('rho-reset', 'n_clicks'),
            Input('mu-reset', 'n_clicks'),
            Input('hp-reset', 'n_clicks'),
            Input('ra-reset', 'n_clicks'),
            Input('kappa-reset', 'n_clicks'),
            Input('Qsc-reset', 'n_clicks'),
            Input('pa-reset', 'n_clicks'),
            Input('pc-reset', 'n_clicks'),
            Input('ps-reset', 'n_clicks'),
            Input('ha-min-reset', 'n_clicks'),
            Input('ha-max-reset', 'n_clicks'),
            Input('nr-reset', 'n_clicks'),
            Input('ha-n-reset', 'n_clicks'),
            Input('c-reset', 'n_clicks'),
            Input('psi-reset', 'n_clicks'),
            Input('case-select', 'value')
        ],
        prevent_initial_call=True
    )
    
    def reset_values(reset_all, *args):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Fetch defaults based on current case selection
        case = args[-1]
        default_bearing = get_default_bearing(case)
        
        # Current values dictionary with proper unit conversions
        current_values = {
            'rho': default_bearing.rho,
            'mu': default_bearing.mu,
            'hp': default_bearing.hp * 1e3,  # Convert to mm
            'ra': default_bearing.ra * 1e3,  # Convert to mm
            'kappa': default_bearing.kappa,
            'Qsc': default_bearing.Qsc,
            'pa': default_bearing.pa * 1e-6,  # Convert to MPa
            'pc': default_bearing.pc * 1e-6,  # Convert to MPa
            'ps': default_bearing.ps * 1e-6,  # Convert to MPa
            'ha_min': default_bearing.ha_min * 1e6,  # Convert to μm
            'ha_max': default_bearing.ha_max * 1e6,  # Convert to μm
            'nr': default_bearing.nr,
            'ha_n': default_bearing.n_ha,
            'c': default_bearing.c * 1e6,
            'psi': default_bearing.Psi
        }
        
        # Reset all values when Reset All clicked OR when case changes
        if button_id in ['reset-all', 'case-select']:
            return list(current_values.values())
        
        # For individual reset buttons
        param = button_id.replace('-reset', '')
        return [
            current_values[p] if p == param else dash.no_update
            for p in ['rho', 'mu', 'hp', 'ra', 'kappa', 'Qsc', 
                      'pa', 'pc', 'ps', 'ha_min', 'ha_max', 'nr', 'ha_n', 'c', 'psi']
        ]

