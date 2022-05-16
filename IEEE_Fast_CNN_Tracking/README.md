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

It is documented in the MAIN_fast_cnn_tracking.ipynb Jupyter notebook. The notebook is to be executed sequentially

## Link to Article

Under review, to be published ...

## Link to data

The saved_data is synthetically generated. If you want to reproduce the exact same results from the paper, please, download it from [Google Drive](https://drive.google.com/file/d/1Us3L7G2hBtgRcqkB1oElKSDdHvJjQo_S/view?usp=sharing). The images from neg_imgs and neg_imgs_tests are also available in the Drive, but they are just random background figures, which have no fiducial markers.

## Bibtex

```
@Article{fast_tracking_melo2022,
    author={Melo, Gabriel Adriano
        and Maximo, Marcos Ricardo Omena de Albuquerque Maximo
        and de Castro, Paulo Andre Lima},
    title={High Speed Marker Tracking for Flight Tests},
    journal={TO BE PUBLISHED},
    year={2022},
    volume={TO BE PUBLISHED},
    number={TO BE PUBLISHED},
    pages={TO BE PUBLISHED},
    doi={TO BE PUBLISHED},
    url={TO BE PUBLISHED}
}
```

