# %% [markdown]
# # TP 4 - Particle Filtering
# 
# ### Le Guillouzic Maël

# %%
import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2 as cv

# %% [markdown]
# You should submit before the deadline a **TP_ParticleFilter_firstname1-surname1_firstname2-surname2.ipynb** containing the completed .ipynb notebook.
# 
# You do not need to attach the data nor the outputs (images and videos) as we must be able to regenerate the output by running your notebook.

# %% [markdown]
# # Particle filter for color-based object tracking

# %% [markdown]
# In this practical work, you will implement a particle filter-based single object tracking approach based on the "color-based probabilistic tracking" framework discussed in the lecture slides (starting from slide 16). The notebook guides you to implement the appearance model, the state propagation model, the particle filter itself, to finally run the tracker on videos.

# %% [markdown]
# ### Extracting the template patch

# %% [markdown]
# We will first use the example video 'data/cows.avi'. The first frame in the video is saved in 'data/cows_first_frame.png' and we load it first below:

# %%
data_dir = 'data'

# %%
zeroth_frame_filename = os.path.join(data_dir, 'cows_first_frame.png')
zeroth_frame = cv.imread(zeroth_frame_filename)

# %%
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal') 
ax.imshow(cv.cvtColor(zeroth_frame, cv.COLOR_BGR2RGB))     
plt.grid(False)
plt.axis('off')
plt.show()

# %% [markdown]
# The single object tracker tracks an object manually defined by a template patch that we have to extract from the first frame. Below I have defined a bounding box width, height, center coordinates $x,y$ so as to track the head of the brown cow. You can change the bounding box parameters to track another object if you want. To help place the bounding box, you can visualize the bounding box on top of the frame in the plot below.

# %%
# Template patch
template_W = 100
template_H = 100
template_x = 600
template_y = 430

# Bounding box in [x1,y1,x2,y2] format:
template_bb = np.array([template_x-template_W//2,template_y-template_H//2,
                        template_x+template_W//2,template_y+template_H//2])

# %%
fig = plt.figure(figsize=(10, 10*zeroth_frame.shape[0]/zeroth_frame.shape[1]))
ax = fig.add_subplot(111, aspect='equal')   
ax.imshow(cv.cvtColor(zeroth_frame, cv.COLOR_BGR2RGB))   
plt.grid(False)
plt.axis('off')

d = np.array([template_x-template_W//2,template_y-template_H//2,template_W,template_H]).astype(np.int32)
ax.add_patch(patches.Rectangle((d[0],d[1]),d[2],d[3],fill=False,lw=3,ec=np.random.rand(1, 3)))
plt.show()

# %% [markdown]
# We will now implement a function that takes a frame (image) as input as well as a bounding box, and extracts a patch from the image at the specified bounding box location. The patch has dimension N along the $x$-axis and M along the $y$-axis. This routine will be used both to extract the template patch, and to extract candidate patches during the tracking.

# %% [markdown]
# The patch pixel values are probed at coordinates regularly spaced between the bounding box corners. This means that the coordinates may fall in-between pixels (they are real valued). Hence you will have to linearly interpolate image values to obtain the patch values.<br>
# Moreover, the bounding box may fall partially outside the image, in which case you will have to extrapolate image values, replicating the border values as boundary condition.

# %%
def extract_patch(image, bounding_box, N=64, M=64):
    """
    Extract a rectangular patch from image at location given by bounding_box.
    
    Returns: numpy.array((M,N,C))
        An image of size N*M, created by linearly interpolating pixel values in the
        original image at evenly spaced coordinates starting at the top-left corner
        of bounding box and ending at the bottom right corner of bounding box
        
    Arguments:
    
    image: numpy.array((H,W,C))
    bounding_box: numpy.array((4,)) in [x1,y1,x2,y2] format
    """
    
    # x and y coordinates at which to interpolate
    map_x, map_y = np.meshgrid(np.linspace(bounding_box[0],bounding_box[2],N,dtype='float32'),
                               np.linspace(bounding_box[1],bounding_box[3],M,dtype='float32'))
    
    # Hint: you can use cv.remap, with linear interpolation and border replication
    patch = cv.remap(image, map_x, map_y, interpolation= cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    
    return patch

# %% [markdown]
# Now we can extract the template patch from `zeroth_frame`, thanks to the bounding box `template_bb` you manually defined previously.

# %%
N = 64 # number of pixels in the patch along the x-axis
M = 64 # number of pixels in the patch along the y-axis 

# Extract the template patch
template = extract_patch(zeroth_frame, template_bb, N, M)

# %%
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal') 
ax.imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))     
plt.grid(False)
plt.axis('off')
plt.show()

# %% [markdown]
# ### First appearance model

# %% [markdown]
# For the particle filter, we need to define the data likelihood given a state $p(z|x^{(i)})$. We will explore two variants of appearance models. The first is based on the candidate patch to template Mean Squared Error. We will first simply compute the Mean Squared Error between the candidate patch (CP) color values and the template (T) color values: $$MSE = \frac{1}{NMC} \Vert CP - T \Vert_2^2 , $$

# %% [markdown]
# where $N,M$ are the patch dimensions and $C$ the number of color channels. We then take $\exp(-\frac{\lambda}{2} \cdot MSE)$ to obtain a value that is a probability (up to an unimportant multiplicative factor).

# %%
def mse_likelihood(image, template, bounding_boxes, lbda):
    """
    Evaluates the likelihood p(image|x_i) for x_i=bounding_boxes[i] for all i.
    The likelihood model is based on the patch-to-template Mean Square Error.
    
    Arguments:
    
    image: numpy.array((H,W,C))
    template: numpy.array((M,N,C))
    bounding_boxes: numpy.array((P,4)) where each row is in the format [x1,y1,x2,y2]
    lbda: real value or 1D numpy.array((L,))
    
    Returns: 1D numpy.array((P,)) if lbda is a scalar, numpy.array((L,P)) if lbda is a 1D array
        An array containing the likelihood values for each bounding box and each value of lbda.
    """
    
    # Extract patches delimited by bounding_boxes, where each
    # patch has the same dimensions as the template
    # Store them in a python list:
    patch_list = [extract_patch(image, bounding_box) for bounding_box in bounding_boxes]
    
    # Compute the mean squared error between patch and template intensities, for every patch in the patch_list.
    mse_list = [np.mean((patch - template)**2) for patch in patch_list]
    mse = np.array(mse_list)
    
    # Compute the likelihood values and store them in a numpy array
    if np.isscalar(lbda):
        likelihoods = np.exp(-lbda*mse/2)
    else:
        likelihoods = np.array([np.exp((-l/2)*mse) for l in lbda])
        
    return likelihoods

# %% [markdown]
# Let's visualize the likelihood map that we obtain for different lambda values. Each pixel value in this map represents the likelihood of a candidate patch centered at the given pixel (and of the same height and width as the template). The maps are subsampled by a factor of 10 for speed of execution.

# %% [markdown]
# As $\lambda$ gets larger, the map becomes more peaked around the actual template patch. For accurate object tracking, we will need maps that are quite peaked.

# %%
bbs = [np.array([x-template_W//2,y-template_H//2,
                 x+template_W//2,y+template_H//2,]) for y in np.arange(0, zeroth_frame.shape[0], 10) 
       for x in np.arange(0, zeroth_frame.shape[1], 10)]
bbs = np.array(bbs)

H_lik = zeroth_frame.shape[0]//10
W_lik = zeroth_frame.shape[1]//10
lbdas = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 1.])
likelihoods = mse_likelihood(zeroth_frame, template, bbs, lbdas)

fig, axs = plt.subplots(2, 3, figsize=(15, 7))
axs[0, 0].imshow(likelihoods[0,:].reshape((H_lik,W_lik)), cmap='gray')
axs[0, 0].set_title('λ = {}'.format(lbdas[0]))
axs[0, 1].imshow(likelihoods[1,:].reshape((H_lik,W_lik)), cmap='gray')
axs[0, 1].set_title('λ = {}'.format(lbdas[1]))
axs[0, 2].imshow(likelihoods[2,:].reshape((H_lik,W_lik)), cmap='gray')
axs[0, 2].set_title('λ = {}'.format(lbdas[2]))
axs[1, 0].imshow(likelihoods[3,:].reshape((H_lik,W_lik)), cmap='gray')
axs[1, 0].set_title('λ = {}'.format(lbdas[3]))
axs[1, 1].imshow(likelihoods[4,:].reshape((H_lik,W_lik)), cmap='gray')
axs[1, 1].set_title('λ = {}'.format(lbdas[4]))
axs[1, 2].imshow(likelihoods[5,:].reshape((H_lik,W_lik)), cmap='gray')
axs[1, 2].set_title('λ = {}'.format(lbdas[5]))
plt.show()

# %% [markdown]
# ### Second appearance model

# %% [markdown]
# The second appearance model is based on a comparison of color histograms instead of a direct comparison of pixel values. We will first convert the patches from BGR (the default color space for opencv images) to HSV (Hue-Saturation-Value). We will then compute the joint histogram of the Hue (H) and Saturation (S) channels for the patch (discarding the information from the Value (V) channel, for reduced sensitivity to the lighting conditions). We discretize each channel to `n_bins` distinct values, resulting in an `n_bins`$\times$`n_bins` histogram.

# %% [markdown]
# By default, the histogram records the number of pixel values that fall in each bin. We will normalize it instead to record the probability of each bin (i.e., the sum of bin values should be normalized to 1).

# %% [markdown]
# Let's start by computing the histogram for the template patch:

# %%
n_bins = 10

# Convert from BGR to HSV then compute the Hue Saturation histogram
# Hints: You can use cv.cvtColor, cv.calcHist, cv.normalize instead of reimplementing these routines yourself
# We compute the histogram only based on the first two channels (Hue-Saturation). 
# The range of the hue channel is 0-179 included. The range of the saturation channel is 0-255 included.
template_hsv = cv.cvtColor(template, cv.COLOR_BGR2HSV)
template_hist = cv.calcHist([template_hsv], [0, 1], None, [n_bins, n_bins], [0, 180, 0, 256])
template_hist = cv.normalize(template_hist, template_hist, norm_type=cv.NORM_L1)

# %% [markdown]
#  

# %% [markdown]
# Now, let's define the data likelihood given a state $p(z|x^{(i)})$. The likelihood is based on the Bhattacharyya distance between normalized histograms as defined in slides 17-18.

# %%
def histogram_likelihood(image, template_hist, bounding_boxes, N, M, lbda):
    """
    Evaluates the likelihood p(image|x_i) for x_i=bounding_boxes[i] for all i.
    The likelihood model is based on a histogram Bhattacharyya distance.
    
    Arguments:
    
    image: numpy.array((H,W,C))
    template_hist: numpy.array((n_bins,n_bins)), the template histogram
    bounding_boxes: numpy.array((P,4)) where each row is in the format [x1,y1,x2,y2]
    lbda: real value or 1D numpy.array((L,))
    
    Returns: numpy.array((P,)) if lbda is a scalar, numpy.array((L,P)) if lbda is a 1D array
        An array containing the likelihood values for each bounding box and each value of lbda.
    """
    
    n_bins = template_hist.shape[0]
    
    # Extract patches delimited by bounding_boxes, where each
    # patch has dimensions N, M
    # Store them in a python list:
    patch_list = [extract_patch(image, bounding_box) for bounding_box in bounding_boxes]
    
    # Convert patches from BGR to HSV
    patch_list = [cv.cvtColor(patch, cv.COLOR_BGR2HSV) for patch in patch_list]
    
    # Compute histograms
    histogram_list = [cv.calcHist([patches], [0,1], None, [n_bins, n_bins], [0,180,0,256]) for patches in patch_list] 
    histogram_list = [cv.normalize(hist, hist, norm_type=cv.NORM_L1) for hist in histogram_list]
    
    # Compute the Bhattacharyya distance between patch and template histograms, for every patch in the patch_list.
    # Hint: you can use cv.compareHist rather than code the metric yourself
    distance_list = [cv.compareHist(histogram, template_hist, method=cv.HISTCMP_BHATTACHARYYA) for histogram in histogram_list]
    square_distances = np.array(distance_list)**2
    
    # Compute the likelihood values and store them in a 1D numpy array
    if np.isscalar(lbda):
        likelihoods = np.exp(-lbda/2 * square_distances )
    else:
        likelihoods = np.array([np.exp((-l/2)*square_distances) for l in lbda])
        
    return likelihoods

# %% [markdown]
# Let's visualize the likelihood maps that we obtain for different lambda values. They are quite different at low lambda values from the ones we obtained previously. For accurate object tracking, we will need the maps to be quite peaked.

# %%
bbs = [np.array([x-template_W//2,y-template_H//2,
                 x+template_W//2,y+template_H//2,]) for y in np.arange(0, zeroth_frame.shape[0], 10) 
       for x in np.arange(0, zeroth_frame.shape[1], 10)]
bbs = np.array(bbs)

H_lik = zeroth_frame.shape[0]//10
W_lik = zeroth_frame.shape[1]//10
lbdas = np.array([0.01, 0.1, 1., 10., 100., 1e3])
likelihoods = histogram_likelihood(zeroth_frame, template_hist, bbs, N, M, lbdas)

fig, axs = plt.subplots(2, 3, figsize=(15, 7))
axs[0, 0].imshow(likelihoods[0,:].reshape((H_lik,W_lik)), cmap='gray')
axs[0, 0].set_title('λ = {}'.format(lbdas[0]))
axs[0, 1].imshow(likelihoods[1,:].reshape((H_lik,W_lik)), cmap='gray')
axs[0, 1].set_title('λ = {}'.format(lbdas[1]))
axs[0, 2].imshow(likelihoods[2,:].reshape((H_lik,W_lik)), cmap='gray')
axs[0, 2].set_title('λ = {}'.format(lbdas[2]))
axs[1, 0].imshow(likelihoods[3,:].reshape((H_lik,W_lik)), cmap='gray')
axs[1, 0].set_title('λ = {}'.format(lbdas[3]))
axs[1, 1].imshow(likelihoods[4,:].reshape((H_lik,W_lik)), cmap='gray')
axs[1, 1].set_title('λ = {}'.format(lbdas[4]))
axs[1, 2].imshow(likelihoods[5,:].reshape((H_lik,W_lik)), cmap='gray')
axs[1, 2].set_title('λ = {}'.format(lbdas[5]))
plt.show()

# %% [markdown]
# ### Motion model

# %% [markdown]
# We will use a variant of the constant velocity motion model that is more reactive to changes in velocity. This variant is not a linear-Gaussian model, but this is not a problem for particle tracking!

# %% [markdown]
# The state (the particle) will consist of six values $[x,y,s,\dot{x},\dot{y},\dot{s}]$, where:
# - $x$ is the coordinate along $x$-axis of the center of the candidate bounding box
# - $y$ is the coordinate along $y$-axis of the center of the candidate bounding box
# - $s$ is the "scale" of the bounding box. The scale parameter resizes the width and height of the candidate bounding box to $s$ times the width/height of the template bounding box *i.e*, $h=s\times$template_H and $w=s\times$template_W

# %% [markdown]
# The motion model / state propagation model is as follows.<br>
# We first apply noise to the velocities: $\dot{x}_k=\dot{x}_{k-1}+\epsilon_{x}$, where $\epsilon_x \sim \mathcal{N}(0, \sigma_x^2)$; $\dot{y}_k=\dot{y}_{k-1}+\epsilon_{y}$, where $\epsilon_y \sim \mathcal{N}(0, \sigma_y^2)$; $\dot{s}_k=\dot{s}_{k-1}+\epsilon_{s}$, where $\epsilon_s \sim \mathcal{N}(0, \sigma_s^2)$.<br> 
# We then compute the new $x,y,s$ based on the velocities, assuming the timestep is $\Delta t= 1$: $x_k=x_{k-1}+\dot{x}_k$, $y_k=y_{k-1}+\dot{y}_k$ and $s_k=s_{k-1}+\dot{s}_k$.

# %% [markdown]
# Let's implement this motion model / state propagation model:

# %%
def state_propagation(particles, process_noise_std=np.array([10.,10.,0.01])):
    """
    Apply the state propagation model to the particles.

    particles: numpy.array((n_particles, 6))
    
    Returns: numpy.array((n_particles, 6)), the updated particles
    """
    
    noise = np.random.randn(particles.shape[0], 3) * process_noise_std
    particles[:, 3:6] += noise
    particles[:, 0:3] += particles[:, 3:6]
    
    return particles

# %% [markdown]
# For compatibility between the motion and appearance models, we also need to implement a routine that computes bounding boxes in the format expected by the likelihood functions implemented above ([x1,y1,x2,y2] format) from the particles in the format $[x,y,s,\dot{x},\dot{y},\dot{s}]$:

# %%
def particles_to_bounding_boxes(particles):
    """
    Convert the format in which particles are written to the bounding box
    [x1,y1,x2,y2] format
    
    particles: numpy.array((n_particles, 6))
    
    Returns: numpy.array((n_particles, 4))
    """
    
    bounding_boxes = []
    for particle in particles:
        x = particle[0]
        y = particle[1]
        s = particle[2]
        H = template_H * s
        W = template_W * s
        x1 = x - W//2
        y1 = y - H//2
        x2 = x + W//2
        y2 = y + H//2
        bounding_boxes.append([x1, y1, x2, y2])
    bounding_boxes = np.array(bounding_boxes)

    return bounding_boxes

# %% [markdown]
# ### Particle filter implementation

# %% [markdown]
# You will now implement the particle filter itself. Fill in the missing lines in the cell below. The particle filter is completely generic with respect to the state propagation model and the likelihood model.

# %%
class ParticleFilter(object):
    """
    Implements a particle filter. The implementation is generic with respect to the choice 
    of state propagation model and of measurement model, it can receive arbitrary functions.
    
    Parameters:
    
    dim_x: int
        Number of state variables for the particle filter. For example, if
        you are tracking the position and velocity of an object in one
        dimension, dim_x would be 2.
    n_particles: int
        Number of particles for the particle filter
    tau: float between 0 and 1
        tau*n_particles is the threshold on the effective sample size under which
        the particles are resampled and their weights reset to 1/N. If 0 we will never
        resample (SIS behaviour). If 1 we will resample at each step (SIR behaviour).
        
    Attributes:
    
    particles: numpy.array(n_particles, dim_x)
        Current state estimates recorded by all particles. Each row
        contains the state of a particle
    weights: numpy.array(n_particles)
        Weights of the particles. The weights sum to 1 over all particles.
    forward: function of signature forward(particles) -> particles
        Function that will be called to perform the predict step of the particle filter,
        whereby the forward state propagation model (including drift and noise) is applied to all particles.
        This returns updated particles.
    likelihood: function of signature likelihood(particles, z) -> likelihoods
        Function that will be called to perform the update step of the particle filter,
        whereby the measurement model and the observation z are used to compute likelihood values
        p(z|x_i) where x_i=particles[i]. The update step uses these likelihood values to update the weights.
        
    The particle filter can be used like this:
    
    PF = ParticleFilter(dim_x, n_particles=..., tau=..., forward=..., likelihood=...)
    self.particles = ... # Initialize particles
    
    while new time step:
        PF.resample()
        PF.predict()
        z = read_measurement(...)
        PF.update(z)
        mean_state = PF.state_expectation()
        
    """
    
    def __init__(self, dim_x, n_particles=100, tau=0.5, forward=None, likelihood=None):
        self.dim_x = dim_x
        self.n_particles = n_particles
        self.tau = tau
        
        if forward is None: # default state propagation model x_k+1 = x_k
            forward = lambda particles: particles 
        if likelihood is None: # default likelihood values ignore measurement z and are all constant
            likelihood = lambda particles, z: np.ones((particles.shape[0],)) 
            
        self.forward = forward
        self.likelihood = likelihood
        
        # Default values for the particles and weights
        self.particles = np.zeros((n_particles, dim_x))
        self.weights = np.ones((n_particles,)) / n_particles
        
    def resample(self, tau=None):
        if tau is None:
            tau = self.tau
            
        threshold = tau*self.n_particles
        
        # Compute the effective sample size
        N_eff = 1/np.sum(self.weights**2)
        
        # resample if necessary
        if N_eff <= threshold:
            # Hint: you can use numpy.random.choice
            self.particles = self.particles[np.random.choice(self.n_particles, size=self.n_particles, replace=True, p=self.weights.flatten())]
            self.weights = np.ones((self.n_particles,))/self.n_particles
            
    def predict(self, forward=None):
        """
        Apply the 'forward' model (state propagation model) to the particles
        """
        if forward is None:
            forward = self.forward
            
        self.particles = forward(self.particles)
        
    def update(self, z, likelihood=None):
        """
        Update the weight of the particles using the measurement model define in 'likelihood'
        """
        if likelihood is None:
            likelihood = self.likelihood
        
        # Compute the vector of likelihoods (one likelihood value per particle)
        likelihoods = likelihood(self.particles, z)
        # likelihoods[i] = p(z|x_i) where x_i = particles[i]
        
        # Hint: formula given in slide 24 under '3. Measure'. Don't forget to normalize!
        self.weights = self.weights*likelihoods/np.sum(self.weights*likelihoods)
        
    def state_expectation(self):
        """
        Computes the empirical average of the particles weighted by their weights,
        as an estimate of the mean of the posterior distribution of the state given the observations.
        
        Returns: numpy.array((dim_x),) 
            A 1D array containing the mean state.
        """
        mean = np.sum(self.particles*self.weights[:,np.newaxis], axis=0)/np.sum(self.weights)
        
        return mean

# %% [markdown]
# The actual state propagation model is passed at initialization time or at runtime as a function (argument `forward`) with signature `forward(particles)->particles`. (in the next section)

# %% [markdown]
# The actual likelihood model is passed at initialization time or at runtime as a function (argument `likelihood`) with signature `likelihood(particles, z)->likelihoods`. (in the next section)

# %% [markdown]
# ### Running the single object tracker

# %% [markdown]
# Finally, you have to fill in the missing lines in the code implementing the particle filter-based single object tracker. You have to try the tracker with both appearance models and find suitable parameters each time (lambda, plus you can play with the process noise standard deviations). Is either appearance model more robust to the parameter settings, or yields better results?

# %% [markdown]
# The parts to complete are the following:
# - the initialization: define the `forward` and `likelihood` functions that will be passed as arguments to initialize a `ParticleFilter`; initialize a `ParticleFilter` PF; initialize the particles of PF to a sensible value;
# - the main loop: run the particle filter on each frame (we actually process only every 10th frame for speed reasons)

# %% [markdown]
# As the tracker is running, you will see the current frame being displayed with:
# - all bounding boxes corresponding to the particles, in the first image displayed
# - the bounding box corresponding to the mean state, in the second image displayed

# %% [markdown]
# The outputs of the code can be found in the folder 'output'. Frames with the overlayed bounding box of the tracked object are saved in the 'output/images folder'. A video is also made based on these frames and stored in 'output/video.avi'.

# %%
import os

from IPython.display import clear_output

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

%matplotlib inline

# %%
video_path = os.path.join('data', 'cows.avi')

colour_particle = np.array([1.,1.,0])
colour_mean = np.random.rand(1, 3)
if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('output/images'):
    os.makedirs('output/images')

# Define likelihood and forward functions (cf. format expected in ParticleFilter)
# Hint: you can use lambda expressions to create functions based on xxx_likelihood and state_propagation 
# but with the correct arguments

# Hint: Here is an example of forward and likelihood for you, but you can change some of the parameters and switch to the
# histogram_likelihood
forward = lambda particles: state_propagation(particles, process_noise_std=np.array([10.,10.,0.01]))
likelihood = lambda particles, z: mse_likelihood(z, template, particles_to_bounding_boxes(particles), lbda=0.01)

# Initialize particle filter
PF = ParticleFilter(dim_x=6, n_particles=100, tau=0.5, forward=forward, likelihood=likelihood)

# Initialize particle x, y, s values to match the template location and size
PF.particles[:,0] = template_x
PF.particles[:,1] = template_y
PF.particles[:,2] = 1

count = -1
video_capture = cv.VideoCapture(video_path)

while(True):
    # Read frame
    ret, frame = video_capture.read()
    if(frame is None): break
    
    # Run the particle filter
    count +=1
    if (count % 10) != 0:
        continue
    
    if count > 0: # ignore first template frame
        # Call ParticleFilter resample, predict and update routines:
        PF.resample()
        PF.predict()
        PF.update(frame, likelihood)
    
    # Compute the expectation of the state posterior
    mean_state = PF.state_expectation()
    mean_bb = particles_to_bounding_boxes(np.expand_dims(mean_state, axis=0))
    
    # Plot the frame with all particles' bounding boxes overlayed
    clear_output(wait=True)
    
    fig = plt.figure(figsize=(10, 10*frame.shape[0]/frame.shape[1]))
    ax = fig.add_subplot(111, aspect='equal')   
    ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))   
    plt.grid(False)
    plt.axis('off')
    
    for d in particles_to_bounding_boxes(PF.particles):
        d = d.astype(np.int32)
        ax.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=1,ec=colour_particle))

    plt.show();
    
    # Plot and save frame with the mean state overlayed
    fig = plt.figure(figsize=(10, 10*frame.shape[0]/frame.shape[1]))
    ax = fig.add_subplot(111, aspect='equal')   
    ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))   
    plt.grid(False)
    plt.axis('off')
    
    d = mean_bb[0].astype(np.int32)
    ax.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colour_mean))

    plt.savefig(os.path.join('output', 'images', str(count).zfill(6) + '.png'))
    plt.show();

# %%
# Reload the png's and save as video for better visualization
image_folder = os.path.join('output', 'images')
video_name = os.path.join('output','video.mp4')

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
frame = cv.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv.imread(os.path.join(image_folder, image)))

cv.destroyAllWindows()
video.release()

# %% [markdown]
# # Optional bonus: Tracking results on a second video

# %% [markdown]
# Let's test the approach on a second short clip of a presidential debate. This one is more difficult and the result will not be perfect. Try to find the model and parameters that make the tracking of the hand work as well as possible.

# %% [markdown]
# ### First frame and template patch:

# %%
zeroth_frame_filename = os.path.join(data_dir, 'pres_debate_first_frame.png')
zeroth_frame = cv.imread(zeroth_frame_filename)

# %%
zeroth_frame.shape

# %%
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal') 
ax.imshow(cv.cvtColor(zeroth_frame, cv.COLOR_BGR2RGB))     
plt.grid(False)
plt.axis('off')
plt.show()

# %%
# Template patch
template_W = 75
template_H = 75
template_x = 570
template_y = 440

# Bounding box in [x1,y1,x2,y2] format:
template_bb = np.array([template_x-template_W//2,template_y-template_H//2,
                        template_x+template_W//2,template_y+template_H//2])

# %%
fig = plt.figure(figsize=(10, 10*zeroth_frame.shape[0]/zeroth_frame.shape[1]))
ax = fig.add_subplot(111, aspect='equal')   
ax.imshow(cv.cvtColor(zeroth_frame, cv.COLOR_BGR2RGB))   
plt.grid(False)
plt.axis('off')

d = np.array([template_x-template_W//2,template_y-template_H//2,template_W,template_H]).astype(np.int32)
ax.add_patch(patches.Rectangle((d[0],d[1]),d[2],d[3],fill=False,lw=3,ec=np.random.rand(1, 3)))
plt.show()

# %%
N = 64 # number of pixels in the patch along the x-axis
M = 64 # number of pixels in the patch along the y-axis 

# Extract the template patch
template = extract_patch(zeroth_frame, template_bb, N, M)

# %%
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal') 
ax.imshow(cv.cvtColor(template, cv.COLOR_BGR2RGB))     
plt.grid(False)
plt.axis('off')
plt.show()

# %% [markdown]
# ### Tracker

# %%
# Define the template histogram if you plan to use the histogram-based likelihood
n_bins = 10

template_hsv = cv.cvtColor(template, cv.COLOR_BGR2HSV)
template_hist = cv.calcHist([template_hsv], [0, 1], None, [n_bins, n_bins], [0, 180, 0, 256])
template_hist = cv.normalize(template_hist, template_hist, norm_type=cv.NORM_L1)

# %%
video_path = os.path.join('data', 'pres_debate.avi')

colour_particle = np.array([1.,1.,0])
colour_mean = np.random.rand(1, 3)
if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('output/images'):
    os.makedirs('output/images')

# Define forward and likelihood functions
forward = lambda particles: state_propagation(particles, process_noise_std=np.array([10.,10.,0.01]))
likelihood = lambda particles, z: histogram_likelihood(z, template_hist, particles_to_bounding_boxes(particles), N, M, lbda=0.01)

# Initialize particle filter
PF = ParticleFilter(dim_x=6, n_particles=100, tau=0.5, forward=forward, likelihood=likelihood)

# Initialize particle x, y, s values to match the template location and size
PF.particles[:,0] = template_x
PF.particles[:,1] = template_y
PF.particles[:,2] = 1

count = -1
video_capture = cv.VideoCapture(video_path)

while(True):
    # Read frame
    ret, frame = video_capture.read()
    if(frame is None): break
    
    # Run the particle filter
    count +=1
    if (count % 1) != 0:
        continue
    
    if count > 0: # ignore first template frame
        # Call ParticleFilter resample, predict and update routines:
        PF.resample()
        PF.predict()
        PF.update(frame, likelihood)
    
    # Compute the expectation of the state posterior
    mean_state = PF.state_expectation()
    mean_bb = particles_to_bounding_boxes(np.expand_dims(mean_state, axis=0))
    
    # Plot the frame with all particles' bounding boxes overlayed
    clear_output(wait=True)
    
    fig = plt.figure(figsize=(10, 10*frame.shape[0]/frame.shape[1]))
    ax = fig.add_subplot(111, aspect='equal')   
    ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))   
    plt.grid(False)
    plt.axis('off')
    
    for d in particles_to_bounding_boxes(PF.particles):
        d = d.astype(np.int32)
        ax.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=1,ec=colour_particle))

    plt.show();
    
    # Plot and save frame with the mean state overlayed
    fig = plt.figure(figsize=(10, 10*frame.shape[0]/frame.shape[1]))
    ax = fig.add_subplot(111, aspect='equal')   
    ax.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))   
    plt.grid(False)
    plt.axis('off')
    
    d = mean_bb[0].astype(np.int32)
    ax.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colour_mean))

    plt.savefig(os.path.join('output', 'images', str(count).zfill(6) + '.png'))
    plt.show();

# %%
# Reload the png's and save as video for better visualization
image_folder = os.path.join('output', 'images')
video_name = os.path.join('output','video.avi')

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
frame = cv.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv.VideoWriter(video_name, 0, 30, (width,height))

for image in images:
    video.write(cv.imread(os.path.join(image_folder, image)))

cv.destroyAllWindows()
video.release()


