# Predictive Maintenance

This repository is intended to enable quick access to datasets for predictive maintenance (PM) tasks (under development).
The following table summarizes the available features,
where the mark \* on dataset names shows
the richness of attributes you may check them up with higher priority.
Note that RUL means remaining useful life.

<!-- :white_check_mark: -->
<!-- :ballot_box_with_check: -->

<center>

| | Timestamp | #Sensor | #Alarm | RUL |　License |
| :--- | :--: | :--: | :--: | :--: | :--- |
| ALPI*     | x |  | 140 |  | CC-BY |
| CBM       | x | 15 | 3 |  | Other |
| CMAPSS    | x | 26 | 2-6 | x | CC0: Public Domain |
| GDD       | x | 5(1) | 3 |  | CC-BY-NC-SA |
| GFD       | x | 4 | 2 |  | CC-BY-SA |
| HydSys*   | x | 17 | 2-4 |  | Other |
| MAPM*     | x | 4 | 5 | x | Other |
| PPD       | x | x | | x | CC-BY-SA |
| UFD       |  | 37-52 | 4 |  | Other |

</center>

<!-- | NASA-B    |  |  |  |  | Other |
| CWRU-B    |  |  |  |  | CC-BY-SA | -->

## Installation

- Python=3.7
- pandas=1.1.2

## Usage

Please put `datasets` directory into your workspace and import it like:

```python
import datasets

# Dataset-specific values will be returned
datasets.ufd.load_data()

# A visualization pdf will be generated
datasets.ufd.gen_summary()
```

Each dataset class has the following functions:
- ```load_data(index)```:  
    Dataset loading specified by 'index'.
    Please see README.md in each dataset directory for more details.
- ```gen_summary(outdir)```:  
    PDF file generation for full dataset visualization.

## Features

### Run-to-Falure

Run-to-Falure data require:
- time column
- event/cencoring column (categorical)
- numerical/categorical feature columns (optional)

## Notebooks

There are Jupyter notebooks for all datasets,
which may help interactive data processing and visualization.


## References

### Introduction to Predictive Maintenance

1. Wikipedia:  
[https://en.wikipedia.org/wiki/Predictive_maintenance](https://en.wikipedia.org/wiki/Predictive_maintenance)
1. Azure AI guide for predictive maintenance solutions:  
[https://docs.microsoft.com/en-us/azure/architecture/data-science-process/predictive-maintenance-playbook](https://docs.microsoft.com/en-us/azure/architecture/data-science-process/predictive-maintenance-playbook)
1. Open source python package for Survival Analysis modeling:  
[https://square.github.io/pysurvival/index.html](https://square.github.io/pysurvival/index.html)
1. Types of proactive maintenance:  
[https://solutions.borderstates.com/types-of-proactive-maintenance/](https://solutions.borderstates.com/types-of-proactive-maintenance/)
1. Common license types for datasets:  
[https://www.kaggle.com/general/116302](https://www.kaggle.com/general/116302)

### Dataset Sources

1. ALPI: Diego Tosato, Davide Dalle Pezze, Chiara Masiero, Gian Antonio Susto, Alessandro Beghi, 2020. Alarm Logs in Packaging Industry (ALPI).  
[https://dx.doi.org/10.21227/nfv6-k750](https://dx.doi.org/10.21227/nfv6-k750)
1. CBM: Condition Based Maintenance of Naval Propulsion Plants Data Set  
[http://archive.ics.uci.edu/ml/datasets/condition+based+maintenance+of+naval+propulsion+plants](http://archive.ics.uci.edu/ml/datasets/condition+based+maintenance+of+naval+propulsion+plants)
1. CMAPSS: NASA Turbofan Jet Engine Data Set:  
[https://www.kaggle.com/behrad3d/nasa-cmaps](https://www.kaggle.com/behrad3d/nasa-cmaps) 
1. GDD: Genesis demonstrator data for machine learning:  
[https://www.kaggle.com/inIT-OWL/genesis-demonstrator-data-for-machine-learning](https://www.kaggle.com/inIT-OWL/genesis-demonstrator-data-for-machine-learning)
1. GFD: Gearbox Fault Diagnosis:  
[https://www.kaggle.com/brjapon/gearbox-fault-diagnosis](https://www.kaggle.com/brjapon/gearbox-fault-diagnosis)
1. HydSys: Predictive Maintenance Of Hydraulics System:  
[https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems](https://archive.ics.uci.edu/ml/datasets/Condition+monitoring+of+hydraulic+systems)
1. MAPM: Microsoft Azure Predictive Maintenance:  
[https://www.kaggle.com/arnabbiswas1/microsoft-azure-predictive-maintenance](https://www.kaggle.com/arnabbiswas1/microsoft-azure-predictive-maintenance)
1. PPD: Production Plant Data for Condition Monitoring:  
[https://www.kaggle.com/inIT-OWL/production-plant-data-for-condition-monitoring](https://www.kaggle.com/inIT-OWL/production-plant-data-for-condition-monitoring)
1. UFD: Ultrasonic flowmeter diagnostics Data Set:  
[https://archive.ics.uci.edu/ml/datasets/Ultrasonic+flowmeter+diagnostics](https://archive.ics.uci.edu/ml/datasets/Ultrasonic+flowmeter+diagnostics)


### TODO

1. Birkl, Christoph. Oxford Battery Degradation Dataset 1. University of Oxford, 2017.  
[https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac)
1. Lu, Jiahuan; Xiong, Rui; Tian, Jinpeng; Wang, Chenxu; Hsu, Chia-Wei; Tsou, Nien-Ti; Sun, Fengchun; Li, Ju (2021), “Battery Degradation Dataset (Fixed Current Profiles＆Arbitrary Uses Profiles)”, Mendeley Data, V2.  
[https://data.mendeley.com/datasets/kw34hhw7xg/2](https://data.mendeley.com/datasets/kw34hhw7xg/2)
1. One Year Industrial Component Degradation  
[https://www.kaggle.com/inIT-OWL/one-year-industrial-component-degradation](https://www.kaggle.com/inIT-OWL/one-year-industrial-component-degradation)
1. Vega shrink-wrapper component degradation  
[https://www.kaggle.com/inIT-OWL/vega-shrinkwrapper-runtofailure-data](https://www.kaggle.com/inIT-OWL/vega-shrinkwrapper-runtofailure-data)
1. NASA Bearing Dataset:  
[https://www.kaggle.com/vinayak123tyagi/bearing-dataset](https://www.kaggle.com/vinayak123tyagi/bearing-dataset)
1. CWRU Bearing Dataset:  
[https://www.kaggle.com/brjapon/cwru-bearing-datasets](https://www.kaggle.com/brjapon/cwru-bearing-datasets)


## License

All the matrials except for datasets is available under MIT lincense.
I preserve all raw data but atatch data loading and preprocessing tools
to each dataset directory so that they are quickly used in Python.
Each dataset should be used under its own lincense.