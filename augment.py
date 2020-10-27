import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


class CTAugment:

    def __init__(self, n_classes, decay=0.99, threshold=0.85, depth=2, n_bins=17):
        self.decay = decay
        self.threshold = threshold
        self.depth = depth
        self.n_bins = n_bins
        self.n_classes = n_classes

        # we need some way of storing functions so that we can randomly sample
        # from them. The format below might not be perfect but it is nice if we
        # can generate a index i wish with which we can access the ith
        # transformation, its corresponding bins and their weights
        self.xforms = []
        self.bins = [[]]
        self.weights = [[]]

        self.AUG_DICT = {
            "autocontrast": {"f": self.autocontrast, "weight": [np.ones(self.n_bins) * 1.0]},
            "blur": {"f": self.blur, "weight": [np.ones(self.n_bins) * 1.0]},
            "brightness": {"f": self.brightness, "weight": [np.ones(self.n_bins) * 1.0]},
            "color": {"f": self.color, "weight": [np.ones(self.n_bins) * 1.0]},
            "contrast": {"f": self.contrast, "weight": [np.ones(self.n_bins) * 1.0]},
            "cutout": {"f": self.cutout, "weight": [np.ones(self.n_bins) * 1.0]},
            "equalize": {"f": self.equalize, "weight": [np.ones(self.n_bins) * 1.0]},
            "invert": {"f": self.invert, "weight": [np.ones(self.n_bins) * 1.0]},
            "identity": {"f": self.identity, "weight": [np.ones(self.n_bins) * 1.0]},
            "posterize": {"f": self.posterize, "weight": [np.ones(self.n_bins) * 1.0]},
            "rescale": {"f": self.rescale, "weight": [np.ones(self.n_bins) * 1.0, np.ones(6) * 1.0]},
            "rotate": {"f": self.rotate, "weight": [np.ones(self.n_bins) * 1.0]},
            "sharpness": {"f": self.sharpness, "weight": [np.ones(self.n_bins) * 1.0]},
            "shear_x": {"f": self.shear_x, "weight": [np.ones(self.n_bins) * 1.0]},
            "shear_y": {"f": self.shear_y, "weight": [np.ones(self.n_bins) * 1.0]},
            "smooth": {"f": self.smooth, "weight": [np.ones(self.n_bins) * 1.0]},
            "solarize": {"f": self.solarize, "weight": [np.ones(self.n_bins) * 1.0]},
            "translate_x": {"f": self.translate_x, "weight": [np.ones(self.n_bins) * 1.0]},
            "translate_y": {"f": self.translate_y, "weight": [np.ones(self.n_bins) * 1.0]}
        }
        self.N = len(self.AUG_DICT.keys())
        self.options = list(self.AUG_DICT.keys())

        self.batch_choices = []
        self.batch_bins = []

    def weight_to_p(self, weight):
        p = weight + (1 - self.decay)  # Avoid to have all zero.
        p = p / p.max()
        p[p < self.threshold] = 0
        return p / np.sum(p)

    def augment(self, x):
        aug_x = Image.fromarray(np.uint8(255 * x))

        choices = [self.options[i] for i in np.random.choice(np.arange(self.N), self.depth, replace=False)]
        bins = []

        for k in range(self.depth):
            choice_key = choices[k]

            transformation = self.AUG_DICT[choice_key]["f"]
            # pick weights for correpsonding function and set weigths to 0 if they
            # are less than 0.8
            w = self.AUG_DICT[choice_key]["weight"][0]
            p = self.weight_to_p(w)
            curr_bins = {}
            curr_bins["bin"] = np.random.choice(np.arange(self.n_bins), p=p)

            if choice_key == "rescale":
                w = self.AUG_DICT[choice_key]["weight"][1]
                p = self.weight_to_p(w)
                curr_bins["bin2"] = np.random.choice(np.arange(6), p=p)

            # we should probably copy here so we do not overwrite original
            aug_x = transformation(aug_x, **curr_bins)
            bins.append(curr_bins)

        return np.array(aug_x), choices, bins

    def augment_batch(self, batch):
        aug_batch = tf.identity(batch)

        # aug_batch = tf.map_fn(aug_batch, self.augment)
        batch_choices = []
        batch_bins = []

        if batch.ndim == 3:
            sample, choices, bins = self.augment(batch)
            batch_choices.append(choices)
            batch_bins.append(bins)
        elif batch.ndim == 4:
            for sample in aug_batch:
                sample, choices, bins = self.augment(sample)
                batch_choices.append(choices)
                batch_bins.append(bins)

        return aug_batch, batch_choices, batch_bins

    def update_weights(self, label, pred, choices, bins, n_classes):
        label_one_hot = np.zeros((label.size, n_classes + 1))
        label_one_hot[np.arange(label.size), label] = 1

        omega = 1 - 1 / (2 * self.n_classes) * np.sum(tf.math.abs(label - pred))

        for k in range(self.depth):

            w = self.AUG_DICT[choices[k]]["weight"][0]
            # tmp = np.copy(w)
            w[bins[k]["bin"]] = self.decay * w[bins[k]["bin"]] + (1 - self.decay) * omega
            # print(tmp-w)
            if choices[k] == "rescale":
                w = self.AUG_DICT[choices[k]]["weight"][1]
                # tmp = np.copy(w)
                w[bins[k]["bin2"]] = self.decay * w[bins[k]["bin2"]] + (1 - self.decay) * omega
                # print(tmp-w)

    def update_weights_batch(self, labels, preds, choices, bins, n_classes):
        [self.update_weights(l, p, c, b, n_classes) for l, p, c, b in zip(labels, preds, choices, bins)]

    def get_param(self, r_min, r_max, bin):
        possible_value = np.linspace(r_min, r_max, self.n_bins)
        return possible_value[bin]

    def autocontrast(self, x, bin):
        param = self.get_param(0, 1, bin)
        return Image.blend(x, ImageOps.autocontrast(x), param)

    def blur(self, x, bin):
        param = self.get_param(0, 1, bin)
        return Image.blend(x, x.filter(ImageFilter.BLUR), param)

    def brightness(self, x, bin):
        param = self.get_param(0, 1, bin)
        return ImageEnhance.Brightness(x).enhance(0.1 + 1.9 * param)

    def color(self, x, bin):
        param = self.get_param(0, 1, bin)
        return ImageEnhance.Color(x).enhance(0.1 + 1.9 * param)

    def contrast(self, x, bin):
        param = self.get_param(0, 1, bin)
        return ImageEnhance.Contrast(x).enhance(0.1 + 1.9 * param)

    def cutout(self, x, bin):
        """Taken directlly from FixMatch code"""
        level = self.get_param(0, 0.5, bin)

        size = 1 + int(level * min(x.size) * 0.499)
        img_height, img_width = x.size
        height_loc = np.random.randint(low=0, high=img_height)
        width_loc = np.random.randint(low=0, high=img_width)
        upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
        lower_coord = (min(img_height, height_loc + size // 2), min(img_width, width_loc + size // 2))
        pixels = x.load()  # create the pixel map
        for i in range(upper_coord[0], lower_coord[0]):  # for every col:
            for j in range(upper_coord[1], lower_coord[1]):  # For every row
                pixels[i, j] = (127, 127, 127)  # set the color accordingly
        return x

    def equalize(self, x, bin):
        param = self.get_param(0, 1, bin)
        return Image.blend(x, ImageOps.equalize(x), param)

    def invert(self, x, bin):
        param = self.get_param(0, 1, bin)
        return Image.blend(x, ImageOps.invert(x), param)

    def identity(self, x, bin):
        return x

    def posterize(self, x, bin):
        param = int(self.get_param(0, 8, bin))
        return ImageOps.posterize(x, param)

    def rescale(self, x, bin, bin2):
        param = self.get_param(0.5, 1, bin)
        methods = (Image.ANTIALIAS, Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.NEAREST)
        method = methods[bin2]
        s = x.size
        scale = param * 0.25
        crop = (scale * s[0], scale * s[1], s[0] * (1 - scale), s[1] * (1 - scale))
        return x.crop(crop).resize(x.size, method)

    def rotate(self, x, bin):
        param = self.get_param(-45, 45, bin)
        angle = int(np.round((2 * param - 1) * 45))
        return x.rotate(angle)

    def sharpness(self, x, bin):
        param = self.get_param(0, 1, bin)
        return ImageEnhance.Sharpness(x).enhance(0.1 + 1.9 * param)

    def shear_x(self, x, bin):
        param = self.get_param(-0.3, 0.3, bin)
        shear = (2 * param - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))

    def shear_y(self, x, bin):
        param = self.get_param(-0.3, 0.3, bin)
        shear = (2 * param - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))

    def smooth(self, x, bin):
        param = self.get_param(0, 1, bin)
        return Image.blend(x, x.filter(ImageFilter.SMOOTH), param)

    def solarize(self, x, bin):
        param = self.get_param(0, 1, bin)
        th = int(param * 255.999)
        return ImageOps.solarize(x, th)

    def translate_x(self, x, bin):
        param = self.get_param(-0.3, 0.3, bin)
        delta = (2 * param - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, 0, delta, 0, 1, 0))

    def translate_y(self, x, bin):
        param = self.get_param(-0.3, 0.3, bin)
        delta = (2 * param - 1) * 0.3
        return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, delta))