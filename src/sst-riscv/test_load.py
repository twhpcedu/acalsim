#!/usr/bin/env python
"""Simple test to verify ACALSim SST component can be loaded"""

import sst

# Try to create an ACALSim component
comp = sst.Component("test_acalsim", "acalsim.ACALSimComponent")
comp.addParams({
    "clock": "1GHz",
    "verbose": "2",
    "max_ticks": "1000"
})

print("SUCCESS: ACALSim component created!")
