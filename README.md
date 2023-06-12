# Code for papers

Code and data for reproducibility.

## IEEE_Fast_CNN_Tracking: High Speed Marker Tracking for Flight Tests

Abstract — Flight testing is a mandatory process to ensure safety
during normal operations and to evaluate an aircraft during its
certification phase. As a test flight may be a high-risk activity that
may result in loss of the aircraft or even loss of life, simulation
models and real-time monitoring systems are crucial to access
the risk and to increase situational awareness and safety. We
propose a new detecting and tracking model based on CNN, that
uses fiducial markers, called HSMT4FT. It is one of the main
components of the Optical Trajectory System (SisTrO) which
is responsible for detecting and tracking fiducial markers in
external stores, in pylons, and in the wings of an aircraft during
Flight Tests. HSMT4FT is a real-time processing model that is
used to measure the trajectory in a store separation test and
even to assess vibrations and wing deflections. Despite the fact
that there are several libraries providing rule-based approaches
for detecting predefined markers, this work contributes by devel-
oping and evaluating three convolutional neural network (CNN)
models for detecting and localizing fiducial markers. We also
compared classical methods for corner detection implemented
in the OpenCV library and the neural network model executed
in the OpenVINO environment. Both the execution time and the
precision/accuracy of those methodologies were evaluated. One of
the CNN models achieved the highest throughput, smaller RMSE,
and highest F1 score among tested and benchmark models.
The best model is fast enough to enable real-time applications
in embedded systems and will be used for real detecting and
tracking in real Flight Tests in the future.

[Open Access IEEE Latin America Transactions link with video and graphical abstract](https://latamt.ieeer9.org/index.php/transactions/article/view/6941)

```
@ARTICLE{fast_tracking_melo,
  author={Melo, Gabriel Adriano and Máximo, Marcos and Castro, Paulo André},
  journal={IEEE Latin America Transactions}, 
  title={High Speed Marker Tracking for Flight Tests}, 
  year={2022},
  volume={20},
  number={10},
  pages={2237-2243},
  doi={10.1109/TLA.2022.9885171},
  url={https://latamt.ieeer9.org/index.php/transactions/article/view/6941}
}
```

## IEEE_vazao_rios: A new approach to river flow forecasting: LSTM and GRU multivariate models

Abstract — Hydroelectric  power  stations  are  responsible  for renewable energy generation, especially in countries with many rivers such as Brazil. It is very important to have good estimates of  the  hydrological  flow  in  order  to  determine  whether thermoelectric  power  plants  should  begin  operation,  an  event that  would  increase  the  costs  of  electricity  and  also  have  a terrible environmental impact. The monthly flow of a river was estimated using two recurrent neural networks techniques: Long-Short Term Memory (LSTM) and Gated Recurrent Unit (GRU). The results were compared with other articles that had the samestructure and used the same data: the Rio Grande river in the Furnas and Camargos dam. 

```tex
@Article{IEEE_melo2022,
    author={Adriano de Melo, Gabriel and Sugimoto, Dylan Nakandakari and Tasinaffo, Paulo Marcelo and Moreira Santos, Afonso Henriques and Cunha, Adilson Marques and Vieira Dias, Luiz Alberto},
    journal={IEEE Latin America Transactions},
    title={A new approach to river flow forecasting: LSTM and GRU multivariate models},
    year={2019},
    volume={17},
    number={12},
    pages={1978-1986},
    url={https://latamt.ieeer9.org/index.php/transactions/article/view/2224/352},
    doi={10.1109/TLA.2019.9011542}
}
```
