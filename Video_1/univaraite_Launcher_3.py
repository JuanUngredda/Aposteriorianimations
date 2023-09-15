from manim import *
from manim.utils.file_ops import open_file as open_media_file
from object_utils import *
import numpy as np
import matplotlib.pyplot as plt


class PointsFromDistribution3(Scene):

    def construct(self):
        group = VGroup()
        x_axis, points_objects = self.Launcher_2_scene(group)
        sample = get_normal_samples(n_samples=50, mu=5, variance=1, seed=0)
        histogram_object = plt.hist(sample, bins=50, range=(0, 10))

        self.play(group.animate.shift(DOWN * 2), group.animate.shift(DOWN * 2), run_time=2)

        func = x_axis.plot(lambda x: self.histogram_evaluator(x, histogram_object))

        rects_right = x_axis.get_riemann_rectangles(
            func,
            x_range=[0, 10],
            dx=0.25,
            input_sample_type="right",
        )
        self.play(FadeOut(points_objects), Create(rects_right), run_time=1.5)
        self.wait(1)

    def histogram_evaluator(self, x, histogram_object):

        pos = histogram_object[1]
        pos_vals = histogram_object[0]

        distance = np.abs(x - pos[:-1])
        idx = np.argmin(distance)
        if distance[idx] > 0.2:
            value = 0
        else:
            value = pos_vals[idx]
        return value

    def Launcher_2_scene(self, group):
        np.random.seed(0)
        # x_axis = NumberLine(x_range=(0, 10), length=10,
        #                     color=BLUE,
        #                     include_numbers=True)
        x_axis = Axes(x_range=(0, 10), color=BLUE, y_axis_config={"color": None})

        group.add(x_axis)
        sample = get_normal_samples(mu=5, variance=1, n_samples=55)
        points_objects = VGroup()
        for i in range(55):
            rand_point = Dot(color=RED)
            rand_point.move_to(x_axis.c2p(sample[i]))
            group.add(rand_point)
            points_objects.add(rand_point)
        self.add(group)

        return x_axis, points_objects


if __name__ == '__main__':
    scene = PointsFromDistribution3()
    scene.render()  # That's it!

    open_media_file(scene.renderer.file_writer.movie_file_path)
