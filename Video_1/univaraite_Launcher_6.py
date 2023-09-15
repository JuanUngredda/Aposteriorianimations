from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np
import matplotlib.pyplot as plt


class PointsFromDistribution6(Scene):

    def construct(self):
        group = VGroup()
        x_axis, rects = self.Launcher_4_scene(group)
        normal_density = x_axis.plot(lambda x: get_normal_pdf(x + 0.2, mu=5, variance=1) * 60 * 0.2)
        self.add(normal_density)
        self.play(FadeOut(rects))

        gamma_func = lambda x: get_gamma_pdf(x, a=3) * 70 * 0.2
        gamma_density = x_axis.plot(gamma_func)
        self.play(Transform(normal_density, gamma_density), run_time=1)
        self.wait(1)

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
        old_group, old_rects_right = self.generate_samples_object(n_samples=50000,
                                                                  group=group,
                                                                  x_axis=x_axis)
        self.add(x_axis, old_group, old_rects_right)

        return x_axis, old_rects_right


if __name__ == '__main__':
    scene = PointsFromDistribution6()
    scene.render()  # That's it!

    open_media_file(scene.renderer.file_writer.movie_file_path)
