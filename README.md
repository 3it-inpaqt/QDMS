A python library that simulates the control of a memristor circuit on a quantum dot.

More precisely, it simulates the voltage output of a memristor circuit composed of a memristor model following your devices. The resistive value is found using pulsed programming on the memristor model. Then, the stability diagram can be generated from the voltage output, which represents the possible control this circuit can have on a quantum dot.

## Installation

QDMS was developed on python 3.9.5, but should work on 3.X.X . To install the library and its dependencies, use:

```
pip install git+https://github.com/3it-nano/QDMS
```

The dependencies are: matlplotlib, numpy and cpickle.


## Uninstallation

To remove the library and all dependencies, use pip-autoremove:
```
pip install pip-autoremove
pip-autoremove qdms -y
pip uninstall pip-autoremove -y
```

To only uninstall the library use:
```
pip uninstall qdms -y
```


## Getting started

To get started with this library, see [Getting Started](https://github.com/3it-nano/QDMS/wiki/Getting-started).

## Credit

Institution : [3IT](https://www.usherbrooke.ca/3it/en/)

Research group : [3IT Nano](https://github.com/3it-nano)

Lead Reseacher : [Pierre-Antoine Mouny](https://github.com/PAMouny)

Programmer : [Sébastien Graveline](https://github.com/Talgarr)
