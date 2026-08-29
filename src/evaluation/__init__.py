"""Final validation, method comparison, and reporting.

Modules
-------
- ``validate`` re-evaluate an optimized configuration with Sionna-RT
- ``compare``  baseline vs. BO vs. MARL, on identical terms
- ``report``   the tilt table: current, optimal, offset
- ``metrics``  surrogate prediction error
- ``analysis`` the spatial maps

Labelling predictions
---------------------
Where a surrogate prediction appears in a report, it is
labelled as such and shown beside the Sionna-RT result, because the gap between them
is itself a result.

Note the division with :mod:`src.kpi`: that package DEFINES the KPIs, this one
validates, compares and presents them. A KPI computed here rather than there
would be a second definition, and the comparison would stop being valid.
"""
