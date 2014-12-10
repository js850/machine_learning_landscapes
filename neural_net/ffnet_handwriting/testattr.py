from pele.storage.database import Minimum
import numpy as np

coords = np.random.random(10)
energy = 10.0

m = Minimum(energy, coords)

setattr(m, "validation_energy", 11.0)
# m.validation_energy = 11.0

print m.validation_energy

def hi(me=10):
    print me

hi(20)