This test data consists of correlators from the QCDSF B meson project.

## LATTICE GENERATION INFORMATION
This collection is called
b5p50kp121040kp120620c2p65__matchtable-4-12104-12062
which means the lattices were generated at beta=5.5,
with kappa_light = 0.12104, and kappa_strange=0.12062
The clover coefficient is 2.65 for light and strange quarks.

The B mesons are generated with either the light or strange
quark properties corresponding to these sea quarks.

## b QUARK GENERATION INFORMATION
7 b quarks are generated for each lattice configuration.
The matchtable-4-12104-12062 in the collection name specifies
which version of the b quark tuning is used for the B mesons.
The properties of these b quarks are written like
matchtable-4-12104_ka0p0716332xi0p7142857cP3p24
which means that for this b quark, kappa=0.0716332, clover=3.24,
and the (chroma) anisotropy is 0.7142857. For tuning, the inverse
of this anisotropy is used.

a confList.txt file is provided, that lists the 7 b quark
names in the order they're required for a tuning analysis.

## CORRELATORS IN DUMPFILES
Each dumpfile (evxpt.res) contains multiple correlators for the
same set of input lattice configurations.
The five correlator types are:

0: B meson, p=0
1: B* meson, p=0
2: B meson, p=1
3: B meson, p=2
4: B meson, p=3
5: B meson, p=0, point sink.

Correlators have smeared source and smeared sink unless otherwise specified.
The source location is automatically set to t=0 for each config, regardless
of the source location in the actual configuration data.

The actual correlators for each lattice configuration are given
between the +RD and -RD tags. For example, the B meson correlators
are given like:

+RD=0
0 5.1e17 -3.0e15
1
....
63 2e15 7e12
0 5.2e17 -5e15
...
-RD

The first column is the time index. The second is the real part of the correlator
the third is the imaginary part of the correlator. In this case, the lattice has
N_t=64, so the index reaches 63 and then returns to 0 for the correlator
from the next lattice configuration. The correlators are given in the order
the lattice configurations appear in the file list.
