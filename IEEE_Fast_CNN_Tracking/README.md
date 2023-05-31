# High Speed Marker Tracking for Flight Tests

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

## Code

It is documented in the fast_cnn_tracking.ipynb Jupyter notebook. The notebook is to be executed sequentially

## Link to Article

[Open Access IEEE Latin America Transactions link with video and graphical abstract](https://latamt.ieeer9.org/index.php/transactions/article/view/6941)

[IEEE Xplore Paywall](https://ieeexplore.ieee.org/document/9885171)

## Link to data

The saved_data is synthetically generated. If you want to reproduce the exact same results from the paper, please, download it from [Google Drive](https://drive.google.com/file/d/1Us3L7G2hBtgRcqkB1oElKSDdHvJjQo_S/view?usp=sharing). The images from neg_imgs and neg_imgs_tests are also available in the Drive, but they are just random background figures, which have no fiducial markers.

## Bibtex

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
