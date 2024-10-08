![{AD00A239-BC78-4F9E-A71B-E24D3A6C5EE6}](https://github.com/user-attachments/assets/bb54391d-28d4-4f2b-bdbe-4fe2b7cc805e)


# Modelling cultural learning for convex color terms
The model used in the paper uses populations
of neural networks, termed agents. The model created existed of 50 generations,
where in each generation 10 agents learned from the previous. Multiple different
initial color spaces were generated as input for the initial generation of agents.
The results of this paper show that convex color terms do emerge when using
a highly structured color space.


The code base and previous convexity data of the thesis: Cultural evolution of convex color terms and degeneracy.

[Paper](https://dspace.uba.uva.nl/server/api/core/bitstreams/f8961ea6-e437-46ab-97ba-faca4d8a98fb/content)
## Replication

[Versions used](requirements.txt)

To replicate results run:


```
python -u ../iteration.py --round_n 1 --epochs 6 --bottleneck 1.0 --iterations 50 --c 1.0 --s 0.0
```











