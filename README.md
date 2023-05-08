# ConDRK
This is the source code for Paper "Contrastive Distillation with Regularized Knowledge for Deep Model Compression on Sensor-based Human Activity Recognition".

### Training Steps

1. Prepare the teacher models for knowledge distillation. Run `teacher_train.py` to generate teachers for each dataset.

2. Train compact student with proposed ConDRK, `condrk_main.py`.
