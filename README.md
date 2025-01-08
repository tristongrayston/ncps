<div align="center"><img src="https://raw.githubusercontent.com/mlech26l/ncps/master/docs/img/banner.png" width="800"/></div>

# CUCAI Project: CFC Networks as Expressive World Models for Robot Control

## 📜 Papers

[Neural Circuit Policies Enabling Auditable Autonomy (Open Access)](https://publik.tuwien.ac.at/files/publik_292280.pdf).  
[Closed-form continuous-time neural networks (Open Access)](https://www.nature.com/articles/s42256-022-00556-7)


This fork is intended to explore the use of specifically CFC networks as more expressive world models. Run as a UVicAI club project, lead by @tristongrayston.

Message on Discord (@triston_g) or email uvicaiclub@gmail.com if you're interested in helping with this project. 

## Todo list: 

### CFC side:


### PPO side:

Graphical output on PPO baseline training could be better.

Running multiple simulations at a time, or optimizing code such that it runs faster on GPU. 

Various tuning to PPO could be better. Maybe playing around with no shared params (does it even make sense to have shared params on non-pixelated inputs?) 

Implement alternative baseline for different environment (half cheetah ideally)





```bib
@article{lechner2020neural,
  title={Neural circuit policies enabling auditable autonomy},
  author={Lechner, Mathias and Hasani, Ramin and Amini, Alexander and Henzinger, Thomas A and Rus, Daniela and Grosu, Radu},
  journal={Nature Machine Intelligence},
  volume={2},
  number={10},
  pages={642--652},
  year={2020},
  publisher={Nature Publishing Group}
}
```
