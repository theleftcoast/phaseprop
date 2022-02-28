"""Pure component parameters."""

from comp import Comp
import refs
from utility import Const


methane = Comp(name='Methane',
               cas_no='74-82-8',
               formula='CH4',
               family='alkane',
               mw=Const(value=16.0425,
                        unit='g/mol',
                        source=refs.dippr),
               tc=Const(value=190.564,
                        unit='K',
                        source=refs.dippr),
               pc=Const(value=4599000.0,
                        unit='Pa',
                        source=refs.dippr))


# methane.formula = "CH4"
# methane.family = "Alkane"
# methane.cas_no = "74-82-8"
# methane.mw = 16.0425
# methane.tc = 190.564
# methane.pc = 4599000.0
# methane.vc = 0.0000986
# methane.acentric = 0.0115478

