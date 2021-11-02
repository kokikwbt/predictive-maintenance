# Predictive Maintenance (PM)

This repository is intended to enable quick access to datasets for predictive maintenance tasks.
The follwoing table summrizes the available features for the PM tasks,
where the marks show:
- `x`: satisfying availablity
- `u`: univariate features
- `m`: multivariate features

The mark \* on dataset names shows
the richness of attributes you may check them up with higher priority.

<center>

|      | Time | Sensors | Alarms  |  RUL |　License |
| :--- | :--: | :--: | :--: | :--: | :--- |
| ALPI*          | x |   | m | x | CC-BY |
| UFD           |   | m | m |  | Other |
| NASA-B        |  |  |  |  | Other |
| CWRU-B        |  |  |  |  | CC-BY-SA |
| MAPM*          | x | m | m | x | Other |
| HydSys*        | x | m | m | x | Other |
| PHM           |  |  |  |  | CC-BY |
| GFD           | x | m | m |  | CC-BY-SA |
| PPD          | x| m | u | x | CC-BY-SA |
| GDD           |  |  |  |  | CC-BY-NC-SA |

</center>


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
1. Common license types for datasets  
[https://www.kaggle.com/general/116302](https://www.kaggle.com/general/116302)

### Dataset Sources

1. ALPI: Diego Tosato, Davide Dalle Pezze, Chiara Masiero, Gian Antonio Susto, Alessandro Beghi, 2020. Alarm Logs in Packaging Industry (ALPI).  
[https://dx.doi.org/10.21227/nfv6-k750](https://dx.doi.org/10.21227/nfv6-k750)
1. UFD: Ultrasonic flowmeter diagnostics Data Set  
[https://archive.ics.uci.edu/ml/datasets/Ultrasonic+flowmeter+diagnostics](https://archive.ics.uci.edu/ml/datasets/Ultrasonic+flowmeter+diagnostics)
1. NASA Bearing Dataset  
[https://www.kaggle.com/vinayak123tyagi/bearing-dataset](https://www.kaggle.com/vinayak123tyagi/bearing-dataset)
1. CWRU Bearing Dataset  
[https://www.kaggle.com/brjapon/cwru-bearing-datasets](https://www.kaggle.com/brjapon/cwru-bearing-datasets)
1. MAPM: Microsoft Azure Predictive Maintenance  
[https://www.kaggle.com/arnabbiswas1/microsoft-azure-predictive-maintenance](https://www.kaggle.com/arnabbiswas1/microsoft-azure-predictive-maintenance)
1. HydSys: Predictive Maintenance Of Hydraulics System  
[https://www.kaggle.com/mayank1897/condition-monitoring-of-hydraulic-systems](https://www.kaggle.com/mayank1897/condition-monitoring-of-hydraulic-systems)
1. GFD: Gearbox Fault Diagnosis  
[https://www.kaggle.com/brjapon/gearbox-fault-diagnosis](https://www.kaggle.com/brjapon/gearbox-fault-diagnosis)
1. PPD: Production Plant Data for Condition Monitoring  
[https://www.kaggle.com/inIT-OWL/production-plant-data-for-condition-monitoring](https://www.kaggle.com/inIT-OWL/production-plant-data-for-condition-monitoring)
1. GDD: Genesis demonstrator data for machine learning  
[https://www.kaggle.com/inIT-OWL/genesis-demonstrator-data-for-machine-learning](https://www.kaggle.com/inIT-OWL/genesis-demonstrator-data-for-machine-learning)

<!-- 1. Condition Based Maintenance (CBM) of Naval Propulsion Plants Data Set  
[http://archive.ics.uci.edu/ml/datasets/condition+based+maintenance+of+naval+propulsion+plants](http://archive.ics.uci.edu/ml/datasets/condition+based+maintenance+of+naval+propulsion+plants) -->


## License

All the matrials except for datasets is available under MIT lincense.
I preserve all raw data but atatch data loading and preprocessing tools
to each dataset directory so that they are quickly used in Python.
Each dataset should be used under its own lincense.