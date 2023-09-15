from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np
import matplotlib.pyplot as plt


class PointsFromDistribution7(Scene):

    def construct(self):
        group = VGroup()
        x_axis = self.Launcher_4_scene(group)

        gamma_func = lambda x: get_gamma_pdf(x, a=3) * 70 * 0.2
        gamma_density = x_axis.plot(gamma_func, color=BLUE)
        self.add(x_axis, gamma_density)

        x_coord = []
        y_coord = []
        z_coord = []
        vlinesgroup = VGroup()
        for x in np.linspace(0, 10, 20):
            x_val = gamma_func(x)
            x_coord.append(x)
            y_coord.append(x_val)
            z_coord.append(0)

            vertical_line = Line(
                start=x_axis.c2p(x, 0),  # Starting point (x, 0)
                end=x_axis.c2p(x, x_val),  # Ending point (x, 5)
                color=BLUE,
            )
            vlinesgroup.add(vertical_line)

        plot = x_axis.plot_line_graph(x_coord, y_coord, z_coord , vertex_dot_style={"color": BLUE})
        self.play(FadeOut(gamma_density), FadeIn(plot["vertex_dots"]), FadeIn(vlinesgroup))
        self.wait(1)
        self.play(FadeOut(plot["vertex_dots"]), FadeOut(vlinesgroup), FadeIn(gamma_density))
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
        # old_group, old_rects_right = self.generate_samples_object(n_samples=50000,
        #                                                           group=group,
        #                                                           x_axis=x_axis)
        # self.add(x_axis, old_group, old_rects_right)

        return x_axis


if __name__ == '__main__':
    scene = PointsFromDistribution7()
    scene.render()  # That's it!

    open_media_file(scene.renderer.file_writer.movie_file_path)
