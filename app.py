import dash

from layouts import create_layout
from callbacks import register_callbacks
from airbearings import (
    AxisymmetricBearing,
    solve_bearing,
    ANALYTIC
)

# Define input fields with default values 
default_bearing = AxisymmetricBearing()

# Initialize bearing and calculate performance
bearing = AxisymmetricBearing()
result = solve_bearing(bearing, soltype=ANALYTIC)

app = dash.Dash(
    __name__,
    meta_tags=[{
        'name': 'viewport',
        'content': 'width=device-width, initial-scale=1.0'
    }],
    title='OpenAirBearing',
    update_title="Loading...",
    assets_folder='assets'
)

# Update app.layout before app.run_server(debug=True):
app.layout = create_layout(default_bearing, bearing, result)

# Register callbacks
register_callbacks(app)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="shortcut icon" href="/assets/favicon.ico">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            <p> Open Air Bearing is available as open-source software under MIT license: <a href="https://github.com/Aalto-Arotor/openAirBearing">github.com/Aalto-Arotor/openAirBearing</a> - Contact: <a href="mailto:mikael.miettinen@iki.fi">mikael.miettinen@iki.fi</a></p>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':   
    app.run_server(debug=True)
