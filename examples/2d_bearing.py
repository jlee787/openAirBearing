import numpy as np
import openairbearing as ab

if __name__ == "__main__":

    # initialize circular bearing with parameters
    bearing = ab.CircularBearing(
        xa=40e-3,
        ya=2 * np.pi,
        Qsc=5,
        nx=50,
        ny=20,
        nh=60,
        error_type="quadratic",
        error=-2e-6,
    )

    # plot bearing shape
    figure = ab.plot_bearing_shape(bearing)
    figure.show()

    # # solve with analytic and numeric methods
    # result = [
    #     ab.solve_bearing(bearing, "analytic"),
    #     ab.solve_bearing(bearing, "numeric2d"),
    # ]

    # # plot results
    # figure = ab.plot_key_results(bearing, result)
    # figure.show()
