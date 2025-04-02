# OpenAirBearing: Open-Source porous gas bearing analysis

[![Unit tests](https://github.com/Aalto-Arotor/openAirBearing/actions/workflows/unittests.yml/badge.svg)](https://github.com/Aalto-Arotor/openAirBearing/actions/workflows/unittests.yml)
[![Test coverage](https://coveralls.io/repos/github/Aalto-Arotor/openAirBearing/badge.svg?branch=main)](https://coveralls.io/github/Aalto-Arotor/openAirBearing?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Aalto-Arotor/openairbearing/blob/main/LICENSE)

## Introduciton
OpenAirbearing includes a backed to model porous bearing performance and a browser GUI to display the results.
Software can be used online at https://www.openairbearing.com in a limited capability, and in full capability when used locally.

Supported features include:
* Circular thrust bearings
* Annular thrust bearings and seals
* Infinitely long linear bearings and seals
* Rectangular bearings

## Mathematical modeling
Analytical and numerical solutions of the Reynolds equation in one dimension fort most common porous gas bearing and seal geometries.
Analytical solutions assume ideal geometry, numerical (finite difference method) solutions consider uneven gap height, uneven permeability, and slip at porous-gap interface.
Implements models from textbooks [1,2] and research of Miettinen et al. [3,4].

## Contact
Main developer is Mikael Miettinen from Arotor lab at Aalto University Finland.

https://www.aalto.fi/en/department-of-energy-and-mechanical-engineering/aalto-arotor-lab

For any questions regarding the software please contact mikael.miettinen@aalto.fi

### References
[1] V. N. Constantinescu, Gas Lubrication, American Society of Mechanical Engineers, 1969. URL: https://archive.org/details/gaslubrication0000cons/

[2] F. Al-Bender, Air Bearings - Theory, Design and Applications, John Wiley &Sons, 2021. doi: https://doi.org/10.1002/9781118926444

[3] M. Miettinen, V. Vainio, R. Theska, R. Viitala, On the static performance of aerostatic elements, Precision Engineering 89 (2024) 1–10. doi:  https://doi.org/10.1016/j.precisioneng.2024.05.017.

[4] M. Miettinen, V. Vainio, R. Viitala, Aerostatic porous annular thrust bearings as seals, Tribology International 200 (2024) 110073. doi: https://doi.org/10.1016/j.triboint.2024.110073.
