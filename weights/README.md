# Pretrained weights
Pretrained weights from different training runs.  
Not tested, but unless there's a bug latest is strongest.

3 different training runs :
- t0 : 3 residual blocks, 64 filters. This is the most solid run for now (writing before t2 finished).  
Pretrained with supervised training on greedy vs greedy games (1 million samples).  
The main differences with t1 are using Adam as an optimizer in both supervised learning and self-play, playing with dirichlet noise and a CPUCT value of 2.

- t1 : 3 residual blocks, 64 filters. Longest run, but seems less solid than t0.  
Pretrained with supervised training on greedy vs greedy games (4 million samples).  
It uses SGD as its optimizer in both supervised learning and self-play, which seemed a lot more solid, with less overfitting than Adam.   
Self-play was done with no dirichlet noise and a CPUCT value of 3.
- t2 : 3 residual blocks, 64 filters. Combination of both t0 and t1.  
Pretrained with supervised training on greedy vs greedy games (4 million samples).  
It uses SGD as its optimizer in both supervised learning and self-play, same config as t1.  
Self-play was done with no dirichlet noise and a CPUCT value of 3.
