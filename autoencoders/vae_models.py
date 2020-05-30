import tensorflow as tf
from tensorflow_probability import distributions as tfd


class AE(tf.keras.Model):
    def __init__(self, 
                 *args, latent_dim=None, 
                 inference_net=None, generative_net=None, 
                 **kwargs):
        super(AE, self).__init__()
        # set latent dim whenever using default inference network
        self.latent_dim = latent_dim 
        # TODO: remove dependency on the "MNIST" input_shape
        if inference_net is None:
            try: 
                self.inference_net = tf.keras.Sequential([
                      tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                      tf.keras.layers.Flatten(),
                      tf.keras.layers.Dense(self.latent_dim + self.latent_dim)
                ])
            except:
                raise ValueError('Latent dimension unknown: set latent_dim parameter.')
        else: 
            self.inference_net = inference_net

        if generative_net is None:
            self.generative_net = tf.keras.Sequential([
                  tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                  tf.keras.layers.Dense(units=28*28*1, activation=tf.nn.relu),
                  tf.keras.layers.Reshape(target_shape=(28, 28, 1))
            ])
        else: 
            self.generative_net = generative_net
        
    def encode(self, x, **encoder_kwargs):
        pass 
    
    def decode(self, x, **decoder_kwargs):
        pass
    
    
class VAE(AE):
    def __init__(self, *args, latent_dist=tfd.Normal, 
                 prior_dist=tfd.Normal, decoder_dist=tfd.Normal, **kwargs):
        super(VAE, self).__init__(*args, **kwargs)
        self.latent_dist_ = latent_dist
        self.prior_dist_ = prior_dist
        self.decoder_dist_ = decoder_dist
        self.decoder_eps_scale = kwargs.get('eps_scale', 1.)
         
    def update_latent_dist(self, loc, logscale):
        return self.latent_dist_(loc=loc, scale=tf.exp(logscale))
    
    def update_prior_dist(self, loc, logscale):
        return self.latent_dist_(loc=loc, scale=tf.exp(logscale))
    
    def decoder_dist(self, loc, logscale=None, eps_scale=None): 
        if logscale is None: # fixed scale --> logscale hyperparameter is tuned by setting eps_scale
            logscale = tf.zeros_like(loc)
        if eps_scale is not None: # default setting for hyperparameter
            self.decoder_eps_scale = eps_scale 
        return self.decoder_dist_(loc=loc, scale=eps_scale * tf.exp(logscale))

    def encode(self, x):
        loc, logscale_preactivation = tf.split(
            self.inference_net(x), num_or_size_splits=2, axis=1)
        return loc, logscale_preactivation 

    def decode(self, z, apply_sigmoid=False):
        decoded = self.generative_net(z)
        if apply_sigmoid:
            decoded = tf.sigmoid(decoded)
        return decoded
 
    def __call__(self, x, **call_args):
        loc, logscale = self.encode(x) # logscale because inference net can return negative values 
        latent_dist = self.update_latent_dist(loc, logscale)
        z = latent_dist.sample()
        decoded = self.decode(z, **call_args)
        
        model_outputs = dict(
            z=z, # sampled from posterior dist q(z|x)
            x=x, # input value
            decoded=decoded, # sample from reconstructed distribution
            posterior_params=dict(loc=loc, logscale=logscale) # parameters of the posterior distribution      
        )
        return model_outputs    
       
    
class GaussianVAE(VAE):
    def update_latent_dist(self, loc, logscale):
        return self.latent_dist_(loc=loc, scale=tf.exp(logscale * .5))
    
    def update_prior_dist(self, loc, logscale):
        return self.prior_dist_(loc=loc, scale=tf.exp(logscale * .5))
    
    def decoder_dist(self, loc, logscale=None, eps_scale=None): 
        if logscale is None: # fixed scale --> logscale hyperparameter is tuned by setting eps_scale
            logscale = tf.zeros_like(loc)
        if eps_scale is not None: # default setting for hyperparameter
            self.decoder_eps_scale = eps_scale 
        return self.decoder_dist_(loc=loc, scale=eps_scale * tf.exp(logscale * 0.5))

    def decoder(self, loc, logscale=None, eps_scale=0.1):
        if logscale is None: # fixed scale
            logscale = tf.zeros_like(loc)          
        return self.decoder_dist_(loc=loc, scale=eps_scale * tf.exp(logscale * .5))
   
    def kl_loss(self, model_outputs):
        posterior = self.update_latent_dist(**model_outputs['posterior_params'])
        prior = self.update_prior_dist(
            tf.zeros_like(posterior.loc), tf.zeros_like(posterior.scale))
        return posterior.kl_divergence(prior)

    
class CauchyVAE(VAE):
    def __init__(self, *args, 
                 latent_dist=tfd.Cauchy, 
                 prior_dist=tfd.Cauchy,
                 decoder_dist=tfd.Cauchy, 
                 **kwargs):
        super(CauchyVAE, self).__init__(*args, 
                 latent_dist=latent_dist, 
                 prior_dist=prior_dist, 
                 decoder_dist=decoder_dist, **kwargs)

    @staticmethod
    def kl_divergence(dist1, dist2):
        sum_scales = dist1.scale + dist2.scale
        diff_locations = dist1.loc - dist2.loc
        numerator = sum_scales**2 + diff_locations**2
        res = tf.math.log(numerator) - tf.math.log(4.) - tf.math.log(dist1.scale) - tf.math.log(dist2.scale)
        return res
       
    def kl_loss(self, model_outputs):
        posterior = self.update_latent_dist(**model_outputs['posterior_params'])
        prior = self.update_prior_dist(
            loc=tf.zeros_like(posterior.loc), logscale=tf.zeros_like(posterior.scale))
        return self.kl_divergence(posterior, prior)

    
class StudentTVAE(VAE):
    def __init__(self, degrees_freedom, *args, 
                 latent_dist=tfd.StudentT, 
                 prior_dist=tfd.StudentT,
                 decoder_dist=tfd.StudentT,
                 **kwargs):
        super(StudentTVAE, self).__init__(*args, 
                 latent_dist=latent_dist, 
                 prior_dist=prior_dist, 
                 decoder_dist=decoder_dist, **kwargs)

    @staticmethod
    def kl_divergence(dist1, dist2):
        sum_scales = dist1.scale + dist2.scale
        diff_locations = dist1.loc - dist2.loc
        numerator = sum_scales**2 + diff_locations**2
        res = tf.math.log(numerator) - tf.math.log(4.) - tf.math.log(dist1.scale) - tf.math.log(dist2.scale)
        return res
       
    def kl_loss(self, model_outputs):
        posterior = self.latent_dist(**model_outputs['posterior_params'])
        prior = self.prior_dist(loc=tf.zeros_like(posterior.loc), logscale=tf.ones_like(posterior.scale))
        return self.kl_divergence(posterior, prior)