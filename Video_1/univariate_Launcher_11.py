from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np
import matplotlib.pyplot as plt


class PointsFromDistribution11(Scene):

    def construct(self):
        group = VGroup()
        x_axis = self.Launcher_4_scene(group)

        prop = 0.8
        var1 = 1
        var2 = 0.5
        multimodal_normal_func_1 = lambda x: (prop) * get_normal_pdf(x=x, mu=5, variance=var1) * 70 * 0.2 + (
                1 - prop) * get_normal_pdf(x=x, mu=2, variance=var2) * 70 * 0.2
        multimodal_normal_func_2_density = x_axis.plot(multimodal_normal_func_1, color=BLUE)
        x_vals_normal_1 = get_normal_samples(mu=5, variance=1, n_samples=50000)
        x_vals_normal_2 = get_normal_samples(mu=2, variance=0.5, n_samples=50000)
        mean_2 = np.mean(x_vals_normal_1) * prop + np.mean(x_vals_normal_2) * (1 - prop)

        vertical_line_3 = Arrow(
            buff=0,
            end=x_axis.c2p(mean_2, 0) + DOWN * 0.15,
            start=x_axis.c2p(mean_2, multimodal_normal_func_1(mean_2)),
            color=BLUE,
            tip_shape=ArrowCircleTip
        )

        normal_func = lambda x: get_normal_pdf(x=x, mu=5, variance=1) * 70 * 0.2
        normal_density = x_axis.plot(normal_func, color=BLUE)

        x_vals = get_normal_samples(mu=5, variance=1, n_samples=50000)
        mean = np.mean(x_vals)

        vertical_line = Arrow(
            buff=0,
            end=x_axis.c2p(mean, 0) + DOWN * 0.15,
            start=x_axis.c2p(mean, normal_func(mean)),
            color=BLUE,
            tip_shape=ArrowCircleTip
        )
        self.add(x_axis)
        self.play(Transform(multimodal_normal_func_2_density, normal_density),
                  Transform(vertical_line_3, vertical_line))
        self.wait()

    def generate_samples_object(self, n_samples, group, x_axis, seed=0):
        sample = get_normal_samples(n_samples=n_samples, mu=5, variance=1, seed=seed)
        histogram_object = plt.hist(sample, bins=50, range=(0, 10))
        func = x_axis.plot(lambda x: self.histogram_evaluator(x, histogram_object))
        rects_right = x_axis.get_riemann_rectangles(
            func,
            x_range=[0, 10],
            dx=0.25,
            input_sample_type="right",
        )

        return group, rects_right

    def histogram_evaluator(self, x, histogram_object):

        pos = histogram_object[1]
        pos_vals = histogram_object[0]
        pos_vals = pos_vals / sum(pos_vals) * np.min([60, sum(pos_vals)])
        distance = np.abs(x - pos[:-1])
        idx = np.argmin(distance)
        if distance[idx] > 0.2:
            value = 0
        else:
            value = pos_vals[idx]
        return value

    def Launcher_4_scene(self, group):
        np.random.seed(0)
        x_axis = Axes(x_range=(0, 10), color=BLUE, y_axis_config={"color": None})
        x_axis.shift(DOWN * 2)
        # old_group, old_rects_right = self.generate_samples_object(n_samples=50000,
        #                                                           group=group,
        #                                                           x_axis=x_axis)
        # self.add(x_axis, old_group, old_rects_right)

        return x_axis


if __name__ == '__main__':
    scene = PointsFromDistribution11()
    scene.render()  # That's it!

    open_media_file(scene.renderer.file_writer.movie_file_path)
