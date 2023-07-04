# NeRF
This repo is for the NeRF's concept and sample training code.

# 1. Abstract
- Our algorithm represents a scene using a fully-connected (non-convolutional) deep network, whose ***input is a single continuous 5D coordinate (Spatial location (x, y, z) and viewing direction ($\theta, \phi$)) and whose output is the volume density and view-dependent emitted radiance at that spatial location.*** We synthesize views by querying 5D coordinates along camera rays and use classic volume rendering techniques to project the output colors and ensities into an image.

# 2. Introduction
- We represent a static scene as a continuous 5D function that ***outputs*** ***the radiance*** emitted in each direction ($\theta, \phi$) at each point (x, y, z) in space, and ***a density*** at each point which acts like a differential opacity controlling how much radiance is accumulated by a ray passing through (x, y, z). Our method optimizes a deep fully-connected neural network without any convolutional layers to represent this function by regressing from a single 5D  
coordinate (x, y, z, $\theta, \phi$) to a single volume density and view-dependent RGB color.  

- To render Nerf from a particular viewpoint :    
<img src="https://github.com/WD4715/SLAMPortfolio/assets/117700793/bb314d9a-a8fe-4802-a042-5edbb6f30c03" width="800" height="200" align="center"/>

- Explaination :
  - we ***march camera rays through the scene to generate a sampled set of 3D points***  
  - we ***use those points and their corresponding 2D viewing directions as input to the neural network to produce an output set of colors and densities***  
  - we use classical volume rendering techniques to ***accumulate those colors and densities*** into a 2D image.  

- We find that the basic implementation of optimzing a neural radiance field representation for a complex scene does not converge to a sufficiently high resolution representation and is ineifficient in the required number of samples per carmera ray. We address these issues by tranforming input 5D coordinates with a positional encoding that enables the MLP to represent higher frequency functions, and we propose a hierarchical sampling procedure to reduce the number of queries required to adequately sample this high-frequency scene representation.

- In summary, our technical contributions are :  
    - An approach for ***representing continuous scenes*** with complex geometry and materials as ***5D neural radiance fields***, parameterized as basic MLP networks.  
    - ***A differentiable rendering procedure*** based on classical volume rendering techiques, which we use to optimize these representations from standard RGB images. This includes ***a hierarchical sampling*** strategy to allocate the MLP’s capacity towards space with visible scene content.  
    - ***A positional encoding*** to map each input 5D coordinate into a higher dimensional space, which enables us to successfully optimize neural radiance fields to represent high-frequency scene content.

# 3. Neural Radiance Field Scene Representation
- We represent a continuous scene as a 5D vector-valued function whose input is a 3D location $\mathbb{X} = (x, y, z) $ and 2D viewing direction $(\theta, \phi)$ and whose output is an emitted color $\mathbb{C} = (r, g, b)$ and volume density $\sigma$. we approximate $\mathbb{F}_{\mathbb{\theta}} : (\mathbb{X}, \mathbb{d}) \rightarrow (\mathbb{C}, \sigma)$


<img src="https://github.com/WD4715/SLAMPortfolio/assets/117700793/c77bede0-94ad-4cbe-8069-6094a19cf511" width="800" height="200" align="center"/>

- We encourage the representation to be multiview consistent by restricting the network to predict the volume density $\sigma $ as a function of only the location $\mathbb{X}$, while allowing the RGB color $\mathbb{C}$ to be predicted as a function of boh location and viewing direction.

# 4. Volume Rendering with Radiance Field  

- We ***render the color of any ray passing through the scene using principles from classical volume rendering.*** The volume density $\sigma(\mathbb{X})$ can be interpreted as the differential ***probability of a ray terminating at an infinitesimal particle at location $\mathbb{X}.$*** The expected color $\mathbb{C}(r)$  of camera ray r(t) = o + td with near and far bounds $t_n$  and $t_f$ is :  
    
    $\mathbb{C}(r) = \int_{t_n}^{t_f} T(t)\sigma(r(t))c(r(t), d)dt, where \space T(t) = exp(-\int_{t_n}^t\sigma(r(s))ds$)

  - The function $T(t)$ means the probability that the ray travels from $t_n$ to t without hitting any other particle. Rendering a view from our continuous neural radiance field requires estimating this intergral $\mathbb{C}(r)$ for a camera ray traced through each pixel of te desired virtual camera.
 
- We use numerically approximations :
    
    $\hat{\mathbb{C}}(r) = \sum_{i=1}^{N}T_i(1-exp(-\sigma_{i}\delta_{i}))c_i\space, where \space T_i = exp(-\sum_{j=1}^{i-1}\sigma_i\delta_i)$
    
    where $\delta_i = t_{t+1} - t_i$ is the distance between adjacent samples. This function for calculating $\hat{\mathbb{C}}(r)$
    
    from the set of ($c_i, \space \sigma_i)$ values is trivaially differentiable and reduces to traditional alpha compositing with alpha values $\alpha_i = 1 - exp(-\sigma_i \delta_i)$.

# 5. Optimizing a Neural Radiance Field
We introduce two improvements to enable representing high-resolution complex scenes. The first is a positional encoding of the input coordinates that assists the MLP in representing high-frequency functions, and the second is a hierarchical sampling procedure that allows us to efficiently sample this high-frequency representations.  
- **Positional Encoding**  
    - They additionally show that ***mapping the inputs to a higher dimensional space using high frequency functions*** before passing them to the network ***enables better fitting of data that contains high frequency variation***.  
    - Reformulating $\mathbb{F}_{\theta}$ as a composition of two functions $\mathbb{F}_{\theta} = \mathbb{F}_{\theta}^{'} \circ\space \gamma$, one learned and one not, significantly improves performance. Here $\gamma$  is a mapping from $\mathbb{R}$  into higher dimensional space $\mathbb{R}^{2L}$, and $\mathbb{F}_{\theta}^{'}$ is still simply a regular MLP.  
        
        $\gamma(p) = (sin(2^0\pi p), cos(2^0\pi p), ..., sin(2^{L-1}\pi p), cos(2^{L-1}\pi p))$  
        
        This function is applied separately to each of the three coordinate values in x and to the three components of the Cartesian viewing direction unit vector $d$  
        
    - We use theses functions to map continuous input coordinates into a higher dimensional space to enable our MLP to more easily approximate a higher frequency function.

 
- **Hierachical volume sampling**  
    - Instead of just using a single network to represent the scene, we simultaneously optimize two networks : one **“Coarse”** and the other **“Fine”. We first sample a set of $N_c$  locations using stratified sampling, and evaluate the “Coarse” network at these locations. Given the output of this “Coarse” network, we then proudce a more informed sampling of points along each ray where samples are biased towards the relevant parts of the volume. To do this, we first rewrite the alpha composited color from the coarse network $\hat{\mathbb{C}}_c(r)$ as a weighted sum of all sampled colors $c_i$ along the ray:**

        $\hat{\mathbb{C}}_c(r) = \sum_{i=1}^{N_c}w_ic_i \space, \space w_i = T_i(1-exp(-\sigma_i \delta_i))$
      
    - Normalizing these weights as $\hat{w}_i = \frac{w_i} {\sum_{j=1}^{N_c}w_j}$ produces a piecewise-constant PDF along the ray. We sample a second set of $N_f$ locations from this distribution using inverse transform sampling, evaluate our “fine” network at the union of the first and second set of samples, and compute the final rendered color of the ray $\hat{\mathbb{C}}_f(r)$ but using $N_c + N_f$ samples. ***This procedure allocates more samples to regions we expect to contain visible content***. This addresses a similar goal as ***importance sampling***, but we use the sampled values ***as a nonuniform discretization of the whole integration domain*** rather than treating each sample as an independent probabilistic estimate of the entire integral.  
