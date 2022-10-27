# Metastable Projections

A dependency for Metastable Baselines, that implements the Projection Layers.  

## Public Version
This branch contains the public version of Metastable Projections. This version does not contain the bindings to [ALR's Project ITPAL (private Repo)](https://github.com/ALRhub/ITPAL), which is necessary to perform the KL Projection. It therefore also does not contain a functional KLProjectionLayer. Only Wasserstein- and Frobenius-Projections are supported as a result.  

You can find the private version of this Repo [here](https://git.dominik-roth.eu/dodox/metastable-projections) ([GitHub Mirror](https://github.com/D-o-d-o-x/metastable-projections))

## Installation
Install this repo as a package:
```
pip install -e .
```

## License
Since this Repo contains code from [Stable Baselines 3 by DLR-RM](https://github.com/DLR-RM/stable-baselines3). SB3 is licensed under the [MIT-License](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE).  
This Repo contains code from [boschresearch/trust-region-layers](https://github.com/boschresearch/trust-region-layers) licensed under the [GPL-License](https://github.com/boschresearch/trust-region-layers/blob/main/LICENSE). Such code has been marked as *Stolen from Fabian's Code (Public Version)*.

As a result this code is available only under the same [GPL-License](https://github.com/boschresearch/trust-region-layers/blob/main/LICENSE).
