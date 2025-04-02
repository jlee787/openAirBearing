import matplotlib.pyplot as plt
import openairbearing as ab


if __name__ == "__main__":

    bearing = ab.CircularBearing()
    result = [
        ab.solve_bearing(bearing, ab.ANALYTIC),
        ab.solve_bearing(bearing, ab.NUMERIC),
    ]
    figure = ab.plot_key_results(bearing, result)
    figure.show()
    # bearing = RectangularBearing(nx=5, ny=7, error_type="quadratic", error=1e-6)
    # b = bearing
    # result = solve_bearing(b, NUMERIC)

    # figure = plots.plot_key_results(b, result)
    # # figure = plots.plot_bearing_shape(bearing)
    # figure.show()
