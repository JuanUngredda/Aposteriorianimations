from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np
import matplotlib.pyplot as plt


class PointsFromDistribution13(Scene):

    def construct(self):
        group = VGroup()
        x_axis = self.Launcher_4_scene(group)

        var = 0.8
        normal_func_4 = lambda x: get_normal_pdf(x=x, mu=5, variance=var) * 70 * 0.2
        normal_density_4 = x_axis.plot(normal_func_4, color=BLUE)

        x_var_1 = 5 + 1.6 * np.sqrt(var)
        x_var_2 = 5 - 1.6 * np.sqrt(var)

        vertical_line_4_1 = DashedLine(
            end=x_axis.c2p(x_var_1, 0),
            start=x_axis.c2p(x_var_1, normal_func_4(x_var_1)),
            color=BLUE,
        )

        vertical_line_4_2 = DashedLine(
            end=x_axis.c2p(x_var_2, 0),
            start=x_axis.c2p(x_var_2, normal_func_4(x_var_2)),
            color=BLUE,
        )
        g4 = VGroup(vertical_line_4_1, vertical_line_4_2)
        self.add(x_axis, normal_density_4, g4)
        self.play(FadeOut(g4), FadeOut(normal_density_4))
        self.wait()

        samples = []
        for i in range(400):
            sample = get_multivaraite_normal_samples(mu=np.ones(2) * 5, cov=np.identity(2), n_samples=1)
            samples.append(sample)

        xy_axis = self.get_xy_axis()
        points_group = VGroup()
        for s in samples:
            rand_point = Dot(xy_axis.coords_to_point(s[0], 0), color=RED)
            # rand_point.move_to(xy_axis.c2p(s[0]))
            points_group.add(rand_point)


        self.play(Transform(x_axis, xy_axis))
        self.play(Write(points_group))

        animations = []
        for k, p in enumerate(points_group):
            vertical_line = Line(start=xy_axis.coords_to_point(samples[k][0], 0),
                                 end=xy_axis.coords_to_point(samples[k][0], samples[k][1]),
                                 color=WHITE)
            # Create a point to move along the vertical path
            animations.append(MoveAlongPath(p, vertical_line))

        self.play(*animations)

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

    def get_xy_axis(self):

        xy_axis = Axes(x_range=(0, 10), y_range=(0, 10), color=BLUE)
        # xy_axis.shift(DOWN * 2)

        return xy_axis


if __name__ == '__main__':
    scene = PointsFromDistribution13()
    scene.render()  # That's it!

    open_media_file(scene.renderer.file_writer.movie_file_path)
