# Astronomy Data Analysis Tool (ADAT_co)

ADAT_co identifies stars of common origin using a two-stage analysis approach. The tool applies the HDBSCAN clustering algorithm followed by photometric validation based on isochrone similarity. The code has successfully passed validation at the Institute of Astronomy, RAS, using open Gaia data.

## Introduction

Astronomy Data Analysis Tool (ADAT_co) plays an important role in the exploration of stellar origins. By leveraging advanced techniques such as HDBSCAN clustering and isochrone-based photometric validation, ADAT_co can support astronomers in identifying stars that have a common origin. Understanding of this is vital for various astronomical studies and contributes to our broader comprehension of stellar evolution, galactic dynamics, and the formation of celestial structures.

## Google Colab Support

The code in this repository is accompanied by a requirements.txt file, tailored for use in Google Colab. This setup has been prepared to enhance simplicity and enable a wider audience to easily access and run the code.

## Getting Started

Prerequisites
Make sure to have the necessary dependencies installed. You can do this by running:
pip install -r requirements.txt

## Installation

Clone the repository.
git clone https://github.com/elenasavvina/adat_co.git

Navigate to the project directory.
cd ADAT_co

Install dependencies.
pip install -r requirements.txt

## Usage

To use ADAT_co, run the following command:
python ADAT_co.py

## Data

Gaia data were used for analysis, they are available. Here is the description of the data access ways: https://www.cosmos.esa.int/web/gaia/data-access

## Why Understanding Common Stellar Origin Matters

Understanding that stars share a common origin is fundamental in various astronomical investigations. It provides insights into the evolution of stellar populations, the dynamics of galaxies, and the intricate processes involved in the formation of celestial structures. ADAT_co contributes to this understanding by employing sophisticated algorithms and data analysis techniques to identify and analyze stars with shared origin.

## Contributing

I welcome contributions!

## License

This project is licensed under the Apache License - v 2.0.
