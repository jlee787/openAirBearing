import numpy as np
import openairbearing as ab
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # initialize circular bearing with parameters
    bearing = ab.JournalBearing()

    # print(np.round(bearing.geom*1e6, 2))
    # # plot bearing shape
    # figure = ab.plot_bearing_shape(bearing)
    # figure.show()

    # solve with analytic and numeric methods
    result = ab.solve_bearing(bearing, "numeric2d")

    # plot results
    # figure = ab.plot_key_results(bearing, result)
    # ab.plot_load_capacity(bearing, result).show()
    # ab.plot_stiffness(bearing, result).show()
    # ab.plot_pressure_distribution(bearing, result).show()
    # ab.plot_supply_flow_rate(bearing, result).show()
    # ab.plot_chamber_flow_rate(bearing, result).show()
    # ab.plot_ambient_flow_rate(bearing, result).show()
    ab.plot_key_results(bearing, result).show()
