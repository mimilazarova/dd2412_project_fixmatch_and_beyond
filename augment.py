import tensorflow_probability as tfp
import numpy as np


class CTAugment:
    def __init__(self, decay, threshold, depth, n_bins):
        self.TRANSFORMATIONS = [self.autocontrast, self.blur, self.brightness,
                                self.color, self.contrast, self.cutout, self.equalize,
                                self.invert, self.identity, self.posterize,
                                self.rescale, self.rotate, self.sharpness,
                                self.shear_x, self.shear_y, self.smooth,
                                self.solarize, self.translate_x, self.translate_y]

        self.decay = decay
        self.threshold = threshold
        self.depth = depth
        self.n_bins = n_bins

        # we need some way of storing functions so that we can randomly sample
        # from them. The format below might not be perfect but it is nice if we
        # can generate a index i with which we can access the ith
        # transformation, its corresponding bins and their weights
        self.bins = [[]]
        self.weights = np.ones([len(self.TRANSFORMATIONS), self.n_bins])

    def augment(self, x):
        for k in range(self.depth):
            # we generate a function randomly (here for each sample in batcg)
            i = np.random.randint(0, len(self.TRANSFORMATIONS), size=x.shape[0])

            # pick weights for correpsonding function and set weigths to 0 if they
            # are less than 0.8
            logits_i = self.weights[i]
            threshold_indices = logits_i < self.threshold
            logits_i[threshold_indices] = 0

            # are the outputs really logits?

            # We should probably do this for a whole batch at onces if possible
            dist = tfp.distributions.Categorical(logits_i)
            bin = dist.sample(1)

            # we should probably copy here so we do not overwrite original
            #x = transformation(x)

        return x

    def update_weights(parameter, weights):
        pass


def autocontrast():
    pass

def blur():
    pass

def brightness():
    pass

def color():
    pass

def contrast():
    pass

def cutout():
    pass

def equalize():
    pass

def invert():
    pass

def identity():
    pass

def posterize():
    pass

def rescale():
    pass

def rotate():
    pass

def sharpness():
    pass

def shear_x():
    pass

def shear_y():
    pass

def smooth():
    pass

def solarize():
    pass

def translate_x():
    pass

def translate_y():
    pass
