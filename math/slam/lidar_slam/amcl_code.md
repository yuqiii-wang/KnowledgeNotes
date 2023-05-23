# AMCL Implementation

* Init histogram (basically a kd-tree) setting all bins to empty (bin has no particle)
* For $M<M_{m}$ and $M<M_{max}$, where $M$ means the particle index, generate/select particle $i$ by $\text{probability}\propto \bold{w}_{t-1}^{[i]}$, indicating the previous step $t-1$ particles with the highest weights are most likely selected (weight represents the confidence of a correct particle's pose)
  * motion model update by this time motion $\bold{u}_t$ to the last time state $\bold{x}_{t-1}^{[i]}$ to produce this time motion result $\bold{x}_{t}^{[M]}$
  * measurement model update (such as by scan matching) by this time motion result $\bold{x}_{t}^{[M]}$ with this time observation $\bold{z}_t$ to produce this time measurement $\bold{w}_{t}^{[M]}$, also served as the weights.
  * If $\bold{x}_{t}^{[M]}$ falls into an empty bin $b$, mark the bin non-empty and increase the total number of particle $M_{m}$ by the KLD formula
  * Keep running the loop until that $M<M_{m}$ and $M<M_{max}$ do not satisfy

<div style="display: flex; justify-content: center;">
      <img src="imgs/kld_sampling_mcl.png" width="30%" height="30%" alt="kld_sampling_mcl" />
</div>
</br>

## Particle Filter

### Convergence

