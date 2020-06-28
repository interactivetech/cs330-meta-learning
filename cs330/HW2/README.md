### HW2 (MAML, ProtoNet)

Debugging tips:
* I found `tf.control_dependencies([tf.print([stuff_to_print])])` helpful. 
* Make sure variable shapes are as intended. Sometimes they are not but the 
errors are not exposed due to undesired shape brodcasting. 

#### P1: MAML

```
python run_maml.py --n_way=5 --k_shot=1 --inner_update_lr=0.4 --num_inner_updates=1

Done initializing, starting training.
Iteration 10: pre-inner-loop accuracy: 0.18800, post-inner-loop accuracy: 0.24400
Iteration 10: pre-inner-loop loss: 11.13007, post-inner-loop loss: 9.84608
Iteration 20: pre-inner-loop accuracy: 0.23200, post-inner-loop accuracy: 0.33600
Iteration 20: pre-inner-loop loss: 10.87544, post-inner-loop loss: 9.04380
Iteration 30: pre-inner-loop accuracy: 0.19200, post-inner-loop accuracy: 0.30400
Iteration 30: pre-inner-loop loss: 10.80296, post-inner-loop loss: 8.69369
Iteration 40: pre-inner-loop accuracy: 0.20000, post-inner-loop accuracy: 0.35200
Iteration 40: pre-inner-loop loss: 10.75301, post-inner-loop loss: 8.39145
Iteration 50: pre-inner-loop accuracy: 0.18400, post-inner-loop accuracy: 0.32800
Iteration 50: pre-inner-loop loss: 10.70303, post-inner-loop loss: 8.20221
Meta-validation pre-inner-loop accuracy: 0.20000, meta-validation post-inner-loop accuracy: 0.33600
Iteration 60: pre-inner-loop accuracy: 0.19200, post-inner-loop accuracy: 0.36000
Iteration 60: pre-inner-loop loss: 10.68665, post-inner-loop loss: 8.04934
Iteration 70: pre-inner-loop accuracy: 0.18400, post-inner-loop accuracy: 0.36000
Iteration 70: pre-inner-loop loss: 10.67006, post-inner-loop loss: 7.92943
Iteration 80: pre-inner-loop accuracy: 0.16000, post-inner-loop accuracy: 0.39200
Iteration 80: pre-inner-loop loss: 10.68184, post-inner-loop loss: 7.83263
Iteration 90: pre-inner-loop accuracy: 0.20800, post-inner-loop accuracy: 0.38400
Iteration 90: pre-inner-loop loss: 10.65353, post-inner-loop loss: 7.74306
Iteration 100: pre-inner-loop accuracy: 0.23200, post-inner-loop accuracy: 0.42400
Iteration 100: pre-inner-loop loss: 10.62107, post-inner-loop loss: 7.65560
Meta-validation pre-inner-loop accuracy: 0.17600, meta-validation post-inner-loop accuracy: 0.32800
Iteration 110: pre-inner-loop accuracy: 0.18400, post-inner-loop accuracy: 0.38400
Iteration 110: pre-inner-loop loss: 10.63363, post-inner-loop loss: 7.60041
Iteration 120: pre-inner-loop accuracy: 0.24000, post-inner-loop accuracy: 0.36800
Iteration 120: pre-inner-loop loss: 10.64405, post-inner-loop loss: 7.55280
Iteration 130: pre-inner-loop accuracy: 0.16800, post-inner-loop accuracy: 0.42400
Iteration 130: pre-inner-loop loss: 10.67215, post-inner-loop loss: 7.51420
Iteration 140: pre-inner-loop accuracy: 0.19200, post-inner-loop accuracy: 0.35200
Iteration 140: pre-inner-loop loss: 10.67645, post-inner-loop loss: 7.48197
Iteration 150: pre-inner-loop accuracy: 0.26400, post-inner-loop accuracy: 0.45600
Iteration 150: pre-inner-loop loss: 10.68588, post-inner-loop loss: 7.43011
Meta-validation pre-inner-loop accuracy: 0.18400, meta-validation post-inner-loop accuracy: 0.31200
Iteration 160: pre-inner-loop accuracy: 0.20800, post-inner-loop accuracy: 0.44800
Iteration 160: pre-inner-loop loss: 10.69212, post-inner-loop loss: 7.40118
Iteration 170: pre-inner-loop accuracy: 0.21600, post-inner-loop accuracy: 0.40000
Iteration 170: pre-inner-loop loss: 10.69718, post-inner-loop loss: 7.38144
Iteration 180: pre-inner-loop accuracy: 0.24000, post-inner-loop accuracy: 0.37600
Iteration 180: pre-inner-loop loss: 10.70797, post-inner-loop loss: 7.35588
Iteration 190: pre-inner-loop accuracy: 0.22400, post-inner-loop accuracy: 0.42400
Iteration 190: pre-inner-loop loss: 10.71459, post-inner-loop loss: 7.33433
Iteration 200: pre-inner-loop accuracy: 0.20800, post-inner-loop accuracy: 0.35200
Iteration 200: pre-inner-loop loss: 10.73321, post-inner-loop loss: 7.31885
Meta-validation pre-inner-loop accuracy: 0.14400, meta-validation post-inner-loop accuracy: 0.44800
Iteration 210: pre-inner-loop accuracy: 0.23200, post-inner-loop accuracy: 0.37600
Iteration 210: pre-inner-loop loss: 10.73953, post-inner-loop loss: 7.29320
Iteration 220: pre-inner-loop accuracy: 0.22400, post-inner-loop accuracy: 0.40800
Iteration 220: pre-inner-loop loss: 10.75064, post-inner-loop loss: 7.27230
Iteration 230: pre-inner-loop accuracy: 0.17600, post-inner-loop accuracy: 0.44800
Iteration 230: pre-inner-loop loss: 10.76594, post-inner-loop loss: 7.24415
Iteration 240: pre-inner-loop accuracy: 0.20000, post-inner-loop accuracy: 0.42400
Iteration 240: pre-inner-loop loss: 10.77727, post-inner-loop loss: 7.22807
Iteration 250: pre-inner-loop accuracy: 0.17600, post-inner-loop accuracy: 0.41600
Iteration 250: pre-inner-loop loss: 10.78907, post-inner-loop loss: 7.21912
Meta-validation pre-inner-loop accuracy: 0.19200, meta-validation post-inner-loop accuracy: 0.51200
...
```
#### P2: ProtoNet

* The distance metric in the embedding space is squared L2 norm. 

```
python run_ProtoNet.py ./omniglot_resized/

[epoch 1/20, episode 50/100] => meta-training loss: 2.08362, meta-training acc: 0.40000, meta-val loss: 2.06878, meta-val acc: 0.30000
[epoch 1/20, episode 100/100] => meta-training loss: 1.67181, meta-training acc: 0.52000, meta-val loss: 1.95912, meta-val acc: 0.43000
[epoch 2/20, episode 50/100] => meta-training loss: 1.64366, meta-training acc: 0.56000, meta-val loss: 1.58095, meta-val acc: 0.46000
[epoch 2/20, episode 100/100] => meta-training loss: 1.57585, meta-training acc: 0.53000, meta-val loss: 1.53967, meta-val acc: 0.58000
[epoch 3/20, episode 50/100] => meta-training loss: 1.64770, meta-training acc: 0.42000, meta-val loss: 1.41419, meta-val acc: 0.59000
[epoch 3/20, episode 100/100] => meta-training loss: 1.98902, meta-training acc: 0.62000, meta-val loss: 1.56005, meta-val acc: 0.59000
[epoch 4/20, episode 50/100] => meta-training loss: 1.53020, meta-training acc: 0.57000, meta-val loss: 1.36582, meta-val acc: 0.64000
[epoch 4/20, episode 100/100] => meta-training loss: 0.99991, meta-training acc: 0.74000, meta-val loss: 1.04949, meta-val acc: 0.73000
[epoch 5/20, episode 50/100] => meta-training loss: 1.49275, meta-training acc: 0.66000, meta-val loss: 0.94757, meta-val acc: 0.70000
[epoch 5/20, episode 100/100] => meta-training loss: 1.16001, meta-training acc: 0.73000, meta-val loss: 0.78531, meta-val acc: 0.77000
[epoch 6/20, episode 50/100] => meta-training loss: 1.24551, meta-training acc: 0.66000, meta-val loss: 0.82909, meta-val acc: 0.73000
[epoch 6/20, episode 100/100] => meta-training loss: 0.84203, meta-training acc: 0.79000, meta-val loss: 0.98809, meta-val acc: 0.66000
t[epoch 7/20, episode 50/100] => meta-training loss: 0.74347, meta-training acc: 0.77000, meta-val loss: 1.12559, meta-val acc: 0.67000
[epoch 7/20, episode 100/100] => meta-training loss: 0.94207, meta-training acc: 0.68000, meta-val loss: 0.61763, meta-val acc: 0.80000
[epoch 8/20, episode 50/100] => meta-training loss: 0.78958, meta-training acc: 0.73000, meta-val loss: 0.74214, meta-val acc: 0.79000
[epoch 8/20, episode 100/100] => meta-training loss: 0.96366, meta-training acc: 0.62000, meta-val loss: 0.91379, meta-val acc: 0.79000
[epoch 9/20, episode 50/100] => meta-training loss: 0.67093, meta-training acc: 0.77000, meta-val loss: 0.51047, meta-val acc: 0.83000
[epoch 9/20, episode 100/100] => meta-training loss: 0.91288, meta-training acc: 0.80000, meta-val loss: 0.58408, meta-val acc: 0.83000
[epoch 10/20, episode 50/100] => meta-training loss: 0.81305, meta-training acc: 0.73000, meta-val loss: 0.87870, meta-val acc: 0.74000
[epoch 10/20, episode 100/100] => meta-training loss: 0.94085, meta-training acc: 0.72000, meta-val loss: 0.47530, meta-val acc: 0.87000
[epoch 11/20, episode 50/100] => meta-training loss: 0.48712, meta-training acc: 0.85000, meta-val loss: 1.41358, meta-val acc: 0.66000
[epoch 11/20, episode 100/100] => meta-training loss: 0.44371, meta-training acc: 0.87000, meta-val loss: 0.61791, meta-val acc: 0.78000
[epoch 12/20, episode 50/100] => meta-training loss: 0.94528, meta-training acc: 0.73000, meta-val loss: 0.73448, meta-val acc: 0.79000
[epoch 12/20, episode 100/100] => meta-training loss: 0.63937, meta-training acc: 0.75000, meta-val loss: 0.74186, meta-val acc: 0.76000
[epoch 13/20, episode 50/100] => meta-training loss: 0.43283, meta-training acc: 0.86000, meta-val loss: 0.60853, meta-val acc: 0.79000
[epoch 13/20, episode 100/100] => meta-training loss: 0.73200, meta-training acc: 0.78000, meta-val loss: 0.72220, meta-val acc: 0.78000
[epoch 14/20, episode 50/100] => meta-training loss: 0.76243, meta-training acc: 0.76000, meta-val loss: 0.71205, meta-val acc: 0.79000
[epoch 14/20, episode 100/100] => meta-training loss: 0.40872, meta-training acc: 0.83000, meta-val loss: 0.48297, meta-val acc: 0.82000
[epoch 15/20, episode 50/100] => meta-training loss: 0.55516, meta-training acc: 0.82000, meta-val loss: 0.64184, meta-val acc: 0.76000
[epoch 15/20, episode 100/100] => meta-training loss: 0.57379, meta-training acc: 0.80000, meta-val loss: 0.55678, meta-val acc: 0.83000
[epoch 16/20, episode 50/100] => meta-training loss: 0.51228, meta-training acc: 0.84000, meta-val loss: 0.74506, meta-val acc: 0.78000
[epoch 16/20, episode 100/100] => meta-training loss: 0.61445, meta-training acc: 0.84000, meta-val loss: 0.41731, meta-val acc: 0.93000
[epoch 17/20, episode 50/100] => meta-training loss: 0.72599, meta-training acc: 0.83000, meta-val loss: 0.63155, meta-val acc: 0.83000
[epoch 17/20, episode 100/100] => meta-training loss: 0.57586, meta-training acc: 0.79000, meta-val loss: 0.59992, meta-val acc: 0.79000
[epoch 18/20, episode 50/100] => meta-training loss: 0.80857, meta-training acc: 0.71000, meta-val loss: 0.35565, meta-val acc: 0.86000
[epoch 18/20, episode 100/100] => meta-training loss: 0.60288, meta-training acc: 0.82000, meta-val loss: 0.58732, meta-val acc: 0.83000
[epoch 19/20, episode 50/100] => meta-training loss: 0.48819, meta-training acc: 0.84000, meta-val loss: 0.23424, meta-val acc: 0.94000
[epoch 19/20, episode 100/100] => meta-training loss: 0.37994, meta-training acc: 0.89000, meta-val loss: 0.69497, meta-val acc: 0.84000
[epoch 20/20, episode 50/100] => meta-training loss: 0.55087, meta-training acc: 0.80000, meta-val loss: 0.79762, meta-val acc: 0.80000
[epoch 20/20, episode 100/100] => meta-training loss: 0.55017, meta-training acc: 0.83000, meta-val loss: 0.36019, meta-val acc: 0.89000
Testing...
[meta-test episode 50/1000] => loss: 0.25619, acc: 0.96000
[meta-test episode 100/1000] => loss: 0.17309, acc: 0.97000
[meta-test episode 150/1000] => loss: 0.23043, acc: 0.97000
[meta-test episode 200/1000] => loss: 0.16642, acc: 0.96000
[meta-test episode 250/1000] => loss: 0.29407, acc: 0.94000
[meta-test episode 300/1000] => loss: 0.25179, acc: 0.94000
[meta-test episode 350/1000] => loss: 0.28806, acc: 0.93000
[meta-test episode 400/1000] => loss: 0.33894, acc: 0.91000
[meta-test episode 450/1000] => loss: 0.13387, acc: 0.99000
[meta-test episode 500/1000] => loss: 0.24665, acc: 0.95000
[meta-test episode 550/1000] => loss: 0.20873, acc: 0.96000
[meta-test episode 600/1000] => loss: 0.30456, acc: 0.93000
[meta-test episode 650/1000] => loss: 0.13266, acc: 0.97000
[meta-test episode 700/1000] => loss: 0.22179, acc: 0.96000
[meta-test episode 750/1000] => loss: 0.34827, acc: 0.86000
[meta-test episode 800/1000] => loss: 0.27711, acc: 0.95000
[meta-test episode 850/1000] => loss: 0.25475, acc: 0.96000
[meta-test episode 900/1000] => loss: 0.33284, acc: 0.91000
[meta-test episode 950/1000] => loss: 0.38721, acc: 0.88000
[meta-test episode 1000/1000] => loss: 0.23666, acc: 0.93000
Average Meta-Test Accuracy: 0.93977, Meta-Test Accuracy Std: 0.02924
```